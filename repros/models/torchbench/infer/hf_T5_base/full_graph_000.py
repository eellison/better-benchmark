class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 2048]", arg1_1: "f16[32128, 768]", arg2_1: "f16[768]", arg3_1: "f16[768, 768]", arg4_1: "f16[768, 768]", arg5_1: "f16[768, 768]", arg6_1: "f16[32, 12]", arg7_1: "f16[768, 768]", arg8_1: "f16[768]", arg9_1: "f16[3072, 768]", arg10_1: "f16[768, 3072]", arg11_1: "f16[768]", arg12_1: "f16[768, 768]", arg13_1: "f16[768, 768]", arg14_1: "f16[768, 768]", arg15_1: "f16[768, 768]", arg16_1: "f16[768]", arg17_1: "f16[3072, 768]", arg18_1: "f16[768, 3072]", arg19_1: "f16[768]", arg20_1: "f16[768, 768]", arg21_1: "f16[768, 768]", arg22_1: "f16[768, 768]", arg23_1: "f16[768, 768]", arg24_1: "f16[768]", arg25_1: "f16[3072, 768]", arg26_1: "f16[768, 3072]", arg27_1: "f16[768]", arg28_1: "f16[768, 768]", arg29_1: "f16[768, 768]", arg30_1: "f16[768, 768]", arg31_1: "f16[768, 768]", arg32_1: "f16[768]", arg33_1: "f16[3072, 768]", arg34_1: "f16[768, 3072]", arg35_1: "f16[768]", arg36_1: "f16[768, 768]", arg37_1: "f16[768, 768]", arg38_1: "f16[768, 768]", arg39_1: "f16[768, 768]", arg40_1: "f16[768]", arg41_1: "f16[3072, 768]", arg42_1: "f16[768, 3072]", arg43_1: "f16[768]", arg44_1: "f16[768, 768]", arg45_1: "f16[768, 768]", arg46_1: "f16[768, 768]", arg47_1: "f16[768, 768]", arg48_1: "f16[768]", arg49_1: "f16[3072, 768]", arg50_1: "f16[768, 3072]", arg51_1: "f16[768]", arg52_1: "f16[768, 768]", arg53_1: "f16[768, 768]", arg54_1: "f16[768, 768]", arg55_1: "f16[768, 768]", arg56_1: "f16[768]", arg57_1: "f16[3072, 768]", arg58_1: "f16[768, 3072]", arg59_1: "f16[768]", arg60_1: "f16[768, 768]", arg61_1: "f16[768, 768]", arg62_1: "f16[768, 768]", arg63_1: "f16[768, 768]", arg64_1: "f16[768]", arg65_1: "f16[3072, 768]", arg66_1: "f16[768, 3072]", arg67_1: "f16[768]", arg68_1: "f16[768, 768]", arg69_1: "f16[768, 768]", arg70_1: "f16[768, 768]", arg71_1: "f16[768, 768]", arg72_1: "f16[768]", arg73_1: "f16[3072, 768]", arg74_1: "f16[768, 3072]", arg75_1: "f16[768]", arg76_1: "f16[768, 768]", arg77_1: "f16[768, 768]", arg78_1: "f16[768, 768]", arg79_1: "f16[768, 768]", arg80_1: "f16[768]", arg81_1: "f16[3072, 768]", arg82_1: "f16[768, 3072]", arg83_1: "f16[768]", arg84_1: "f16[768, 768]", arg85_1: "f16[768, 768]", arg86_1: "f16[768, 768]", arg87_1: "f16[768, 768]", arg88_1: "f16[768]", arg89_1: "f16[3072, 768]", arg90_1: "f16[768, 3072]", arg91_1: "f16[768]", arg92_1: "f16[768, 768]", arg93_1: "f16[768, 768]", arg94_1: "f16[768, 768]", arg95_1: "f16[768, 768]", arg96_1: "f16[768]", arg97_1: "f16[3072, 768]", arg98_1: "f16[768, 3072]", arg99_1: "f16[768]", arg100_1: "f16[768]", arg101_1: "f16[768, 768]", arg102_1: "f16[768, 768]", arg103_1: "f16[768, 768]", arg104_1: "f16[32, 12]", arg105_1: "f16[768, 768]", arg106_1: "f16[768]", arg107_1: "f16[768, 768]", arg108_1: "f16[768, 768]", arg109_1: "f16[768, 768]", arg110_1: "f16[768, 768]", arg111_1: "f16[768]", arg112_1: "f16[3072, 768]", arg113_1: "f16[768, 3072]", arg114_1: "f16[768]", arg115_1: "f16[768, 768]", arg116_1: "f16[768, 768]", arg117_1: "f16[768, 768]", arg118_1: "f16[768, 768]", arg119_1: "f16[768]", arg120_1: "f16[768, 768]", arg121_1: "f16[768, 768]", arg122_1: "f16[768, 768]", arg123_1: "f16[768, 768]", arg124_1: "f16[768]", arg125_1: "f16[3072, 768]", arg126_1: "f16[768, 3072]", arg127_1: "f16[768]", arg128_1: "f16[768, 768]", arg129_1: "f16[768, 768]", arg130_1: "f16[768, 768]", arg131_1: "f16[768, 768]", arg132_1: "f16[768]", arg133_1: "f16[768, 768]", arg134_1: "f16[768, 768]", arg135_1: "f16[768, 768]", arg136_1: "f16[768, 768]", arg137_1: "f16[768]", arg138_1: "f16[3072, 768]", arg139_1: "f16[768, 3072]", arg140_1: "f16[768]", arg141_1: "f16[768, 768]", arg142_1: "f16[768, 768]", arg143_1: "f16[768, 768]", arg144_1: "f16[768, 768]", arg145_1: "f16[768]", arg146_1: "f16[768, 768]", arg147_1: "f16[768, 768]", arg148_1: "f16[768, 768]", arg149_1: "f16[768, 768]", arg150_1: "f16[768]", arg151_1: "f16[3072, 768]", arg152_1: "f16[768, 3072]", arg153_1: "f16[768]", arg154_1: "f16[768, 768]", arg155_1: "f16[768, 768]", arg156_1: "f16[768, 768]", arg157_1: "f16[768, 768]", arg158_1: "f16[768]", arg159_1: "f16[768, 768]", arg160_1: "f16[768, 768]", arg161_1: "f16[768, 768]", arg162_1: "f16[768, 768]", arg163_1: "f16[768]", arg164_1: "f16[3072, 768]", arg165_1: "f16[768, 3072]", arg166_1: "f16[768]", arg167_1: "f16[768, 768]", arg168_1: "f16[768, 768]", arg169_1: "f16[768, 768]", arg170_1: "f16[768, 768]", arg171_1: "f16[768]", arg172_1: "f16[768, 768]", arg173_1: "f16[768, 768]", arg174_1: "f16[768, 768]", arg175_1: "f16[768, 768]", arg176_1: "f16[768]", arg177_1: "f16[3072, 768]", arg178_1: "f16[768, 3072]", arg179_1: "f16[768]", arg180_1: "f16[768, 768]", arg181_1: "f16[768, 768]", arg182_1: "f16[768, 768]", arg183_1: "f16[768, 768]", arg184_1: "f16[768]", arg185_1: "f16[768, 768]", arg186_1: "f16[768, 768]", arg187_1: "f16[768, 768]", arg188_1: "f16[768, 768]", arg189_1: "f16[768]", arg190_1: "f16[3072, 768]", arg191_1: "f16[768, 3072]", arg192_1: "f16[768]", arg193_1: "f16[768, 768]", arg194_1: "f16[768, 768]", arg195_1: "f16[768, 768]", arg196_1: "f16[768, 768]", arg197_1: "f16[768]", arg198_1: "f16[768, 768]", arg199_1: "f16[768, 768]", arg200_1: "f16[768, 768]", arg201_1: "f16[768, 768]", arg202_1: "f16[768]", arg203_1: "f16[3072, 768]", arg204_1: "f16[768, 3072]", arg205_1: "f16[768]", arg206_1: "f16[768, 768]", arg207_1: "f16[768, 768]", arg208_1: "f16[768, 768]", arg209_1: "f16[768, 768]", arg210_1: "f16[768]", arg211_1: "f16[768, 768]", arg212_1: "f16[768, 768]", arg213_1: "f16[768, 768]", arg214_1: "f16[768, 768]", arg215_1: "f16[768]", arg216_1: "f16[3072, 768]", arg217_1: "f16[768, 3072]", arg218_1: "f16[768]", arg219_1: "f16[768, 768]", arg220_1: "f16[768, 768]", arg221_1: "f16[768, 768]", arg222_1: "f16[768, 768]", arg223_1: "f16[768]", arg224_1: "f16[768, 768]", arg225_1: "f16[768, 768]", arg226_1: "f16[768, 768]", arg227_1: "f16[768, 768]", arg228_1: "f16[768]", arg229_1: "f16[3072, 768]", arg230_1: "f16[768, 3072]", arg231_1: "f16[768]", arg232_1: "f16[768, 768]", arg233_1: "f16[768, 768]", arg234_1: "f16[768, 768]", arg235_1: "f16[768, 768]", arg236_1: "f16[768]", arg237_1: "f16[768, 768]", arg238_1: "f16[768, 768]", arg239_1: "f16[768, 768]", arg240_1: "f16[768, 768]", arg241_1: "f16[768]", arg242_1: "f16[3072, 768]", arg243_1: "f16[768, 3072]", arg244_1: "f16[768]", arg245_1: "f16[768, 768]", arg246_1: "f16[768, 768]", arg247_1: "f16[768, 768]", arg248_1: "f16[768, 768]", arg249_1: "f16[768]", arg250_1: "f16[768, 768]", arg251_1: "f16[768, 768]", arg252_1: "f16[768, 768]", arg253_1: "f16[768, 768]", arg254_1: "f16[768]", arg255_1: "f16[3072, 768]", arg256_1: "f16[768, 3072]", arg257_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "f16[]" = self._tensor_constant0;  _tensor_constant0 = None
        _tensor_constant1: "f16[]" = self._tensor_constant1;  _tensor_constant1 = None
        _tensor_constant2: "f16[]" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "f16[1, 2048, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_317: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(embedding_2, torch.float32)
        pow_26: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_317, 2);  convert_element_type_317 = None
        mean_25: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_72: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_52: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_25);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_318: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_52, torch.float16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_53: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg100_1, convert_element_type_318);  arg100_1 = convert_element_type_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_290: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_53, [2048, 768])
        permute_133: "f16[768, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_72: "f16[2048, 768]" = torch.ops.aten.mm.default(view_290, permute_133);  view_290 = permute_133 = None
        view_291: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_72, [1, 2048, 768]);  mm_72 = None
        view_292: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_291, [1, 2048, -1, 64]);  view_291 = None
        permute_134: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_51: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_134, [1, 12, 2048, 64]);  permute_134 = None
        view_299: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_51, [12, 2048, 64]);  expand_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_293: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_53, [2048, 768])
        permute_135: "f16[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        mm_73: "f16[2048, 768]" = torch.ops.aten.mm.default(view_293, permute_135);  view_293 = permute_135 = None
        view_294: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_73, [1, 2048, 768]);  mm_73 = None
        view_295: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_294, [1, 2048, -1, 64]);  view_294 = None
        permute_136: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_139: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2]);  permute_136 = None
        expand_52: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_139, [1, 12, 64, 2048]);  permute_139 = None
        view_300: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_52, [12, 64, 2048]);  expand_52 = None
        bmm_24: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_299, view_300);  view_299 = view_300 = None
        view_301: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_24, [1, 12, 2048, 2048]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_15: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_16: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota_15, 0);  iota_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_14: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(iota_14, 1);  iota_14 = None
        add_73: "i64[2048, 1]" = torch.ops.aten.add.Tensor(unsqueeze_15, 0);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_13: "i64[2048, 2048]" = torch.ops.aten.sub.Tensor(unsqueeze_16, add_73);  unsqueeze_16 = add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_55: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[2048, 2048]" = torch.ops.aten.minimum.default(sub_13, full_default_55);  sub_13 = full_default_55 = None
        neg_24: "i64[2048, 2048]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[2048, 2048]" = torch.ops.aten.lt.Scalar(neg_24, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_327: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(neg_24, torch.float32)
        div_14: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_327, 16);  convert_element_type_327 = None
        log_1: "f32[2048, 2048]" = torch.ops.aten.log.default(div_14);  div_14 = None
        div_15: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_54: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(div_15, 16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_328: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(mul_54, torch.int64);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_74: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_328, 16);  convert_element_type_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_56: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[2048, 2048]" = torch.ops.aten.minimum.default(add_74, full_default_56);  add_74 = full_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_28: "i64[2048, 2048]" = torch.ops.aten.where.self(lt_1, neg_24, minimum_2);  lt_1 = neg_24 = minimum_2 = None
        add_75: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(where_28, 0);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f16[2048, 2048, 12]" = torch.ops.aten.embedding.default(arg104_1, add_75);  arg104_1 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_140: "f16[12, 2048, 2048]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "f16[1, 12, 2048, 2048]" = torch.ops.aten.unsqueeze.default(permute_140, 0);  permute_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_69: "i64[2048]" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_69, 0);  add_69 = None
        unsqueeze_10: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_68: "i64[2048]" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_68, 0);  add_68 = None
        unsqueeze_7: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 2048, 2048]" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_49: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(le, [1, -1, 2048, 2048]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_51: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand_49, full_default_51, full_default_52);  expand_49 = full_default_51 = full_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_76: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(unsqueeze_17, where_26);  unsqueeze_17 = where_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_77: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_301, add_76);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_329: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_77, torch.float32);  add_77 = None
        amax_12: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_329, [-1], True)
        sub_14: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_329, amax_12);  convert_element_type_329 = amax_12 = None
        exp_12: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_330: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_16, torch.float16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_330, [1, 12, 2048, 2048]);  convert_element_type_330 = None
        view_304: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_53, [12, 2048, 2048]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_296: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_53, [2048, 768]);  mul_53 = None
        permute_137: "f16[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_74: "f16[2048, 768]" = torch.ops.aten.mm.default(view_296, permute_137);  view_296 = permute_137 = None
        view_297: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_74, [1, 2048, 768]);  mm_74 = None
        view_298: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_297, [1, 2048, -1, 64]);  view_297 = None
        permute_138: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_54: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_138, [1, 12, 2048, 64]);  permute_138 = None
        view_305: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_54, [12, 2048, 64]);  expand_54 = None
        bmm_25: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_304, view_305);  view_304 = view_305 = None
        view_306: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 12, 2048, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_141: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        clone_64: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_307: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_64, [1, 2048, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_308: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_307, [2048, 768]);  view_307 = None
        permute_142: "f16[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_75: "f16[2048, 768]" = torch.ops.aten.mm.default(view_308, permute_142);  view_308 = permute_142 = None
        view_309: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_75, [1, 2048, 768]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_78: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(embedding_2, view_309);  embedding_2 = view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_335: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_24: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_78);  add_78 = None
        any_25: "b8[]" = torch.ops.aten.any.default(isinf_24);  isinf_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_58: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "f32[]" = torch.ops.aten.where.self(any_25, full_default_58, full_default_57);  any_25 = full_default_58 = full_default_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_25: "f32[]" = torch.ops.aten.neg.default(where_29)
        clamp_min_24: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_335, neg_25);  convert_element_type_335 = neg_25 = None
        clamp_max_24: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_24, where_29);  clamp_min_24 = where_29 = None
        convert_element_type_336: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_24, torch.float16);  clamp_max_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_337: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_336, torch.float32)
        pow_27: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_337, 2);  convert_element_type_337 = None
        mean_26: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_55: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_336, rsqrt_26);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_338: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_56: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg106_1, convert_element_type_338);  arg106_1 = convert_element_type_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_310: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_56, [2048, 768]);  mul_56 = None
        permute_143: "f16[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_76: "f16[2048, 768]" = torch.ops.aten.mm.default(view_310, permute_143);  view_310 = permute_143 = None
        view_311: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_76, [1, 2048, 768]);  mm_76 = None
        view_312: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_311, [1, 2048, -1, 64]);  view_311 = None
        permute_144: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_55: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_144, [1, 12, 2048, 64]);  permute_144 = None
        view_319: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_55, [12, 2048, 64]);  expand_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f16[1, 2048, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_1: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_1: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_1, [2048, 768])
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f16[2048, 768]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm, [1, 2048, 768]);  mm = None
        view_3: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_2, [1, 2048, -1, 64]);  view_2 = None
        permute_1: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_1: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_1, [1, 12, 2048, 64]);  permute_1 = None
        view_10: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_1, [12, 2048, 64]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_4: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_1, [2048, 768])
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f16[2048, 768]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_1, [1, 2048, 768]);  mm_1 = None
        view_6: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_5, [1, 2048, -1, 64]);  view_5 = None
        permute_3: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_2: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 2048]);  permute_6 = None
        view_11: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_2, [12, 64, 2048]);  expand_2 = None
        bmm: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm, [1, 12, 2048, 2048]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_5: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(iota_4, 1);  iota_4 = None
        add_3: "i64[2048, 1]" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[2048, 2048]" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:200 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt: "b8[2048, 2048]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_10: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_2: "i64[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 16);  convert_element_type_10 = None
        add_4: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:201 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[2048, 2048]" = torch.ops.aten.abs.default(sub);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[2048, 2048]" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_11: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_11, 8);  convert_element_type_11 = None
        log: "f32[2048, 2048]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_3: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_12: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_12, 8);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_2: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[2048, 2048]" = torch.ops.aten.minimum.default(add_5, full_default_2);  add_5 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[2048, 2048]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f16[2048, 2048, 12]" = torch.ops.aten.embedding.default(arg6_1, add_6);  arg6_1 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f16[12, 2048, 2048]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f16[1, 12, 2048, 2048]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[2048]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 2048, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(ge, [1, -1, 2048, 2048]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_13: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        amax: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_2, torch.float16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_14, [1, 12, 2048, 2048]);  convert_element_type_14 = None
        view_15: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_3, [12, 2048, 2048]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_7: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_1, [2048, 768]);  mul_1 = None
        permute_4: "f16[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "f16[2048, 768]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 2048, 768]);  mm_2 = None
        view_9: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_8, [1, 2048, -1, 64]);  view_8 = None
        permute_5: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_4: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 2048, 64]);  permute_5 = None
        view_16: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 2048, 64]);  expand_4 = None
        bmm_1: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 2048, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_2: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_18: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 2048, -1]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_19: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_18, [2048, 768]);  view_18 = None
        permute_9: "f16[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "f16[2048, 768]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_3, [1, 2048, 768]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_9: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_19: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_9);  add_9 = None
        any_1: "b8[]" = torch.ops.aten.any.default(isinf);  isinf = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[]" = torch.ops.aten.where.self(any_1, full_default_4, full_default_3);  any_1 = full_default_4 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg: "f32[]" = torch.ops.aten.neg.default(where_2)
        clamp_min: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_19, neg);  convert_element_type_19 = neg = None
        clamp_max: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min, where_2);  clamp_min = where_2 = None
        convert_element_type_20: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.float16);  clamp_max = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_21: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32)
        pow_2: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2);  convert_element_type_21 = None
        mean_1: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_4: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, rsqrt_1);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_22: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.float16);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_5: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_22);  arg8_1 = convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_21: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_5, [2048, 768]);  mul_5 = None
        permute_10: "f16[768, 3072]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_4, [1, 2048, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_22);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_23: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu, [2048, 3072]);  relu = None
        permute_11: "f16[3072, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "f16[2048, 768]" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_5, [1, 2048, 768]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_11: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_20, view_24);  convert_element_type_20 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_27: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_1: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_11);  add_11 = None
        any_2: "b8[]" = torch.ops.aten.any.default(isinf_1);  isinf_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[]" = torch.ops.aten.where.self(any_2, full_default_6, full_default_5);  any_2 = full_default_6 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_1: "f32[]" = torch.ops.aten.neg.default(where_3)
        clamp_min_1: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_27, neg_1);  convert_element_type_27 = neg_1 = None
        clamp_max_1: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_1, where_3);  clamp_min_1 = where_3 = None
        convert_element_type_28: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.float16);  clamp_max_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_29: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32)
        pow_3: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 2);  convert_element_type_29 = None
        mean_2: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_6: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, rsqrt_2);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_30: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_7: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg11_1, convert_element_type_30);  arg11_1 = convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_25: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_7, [2048, 768])
        permute_12: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_6: "f16[2048, 768]" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 2048, 768]);  mm_6 = None
        view_27: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_26, [1, 2048, -1, 64]);  view_26 = None
        permute_13: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_5: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_13, [1, 12, 2048, 64]);  permute_13 = None
        view_34: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_5, [12, 2048, 64]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_28: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_7, [2048, 768])
        permute_14: "f16[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "f16[2048, 768]" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
        view_29: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_7, [1, 2048, 768]);  mm_7 = None
        view_30: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_29, [1, 2048, -1, 64]);  view_29 = None
        permute_15: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_18: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_6: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_18, [1, 12, 64, 2048]);  permute_18 = None
        view_35: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_6, [12, 64, 2048]);  expand_6 = None
        bmm_2: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 2048, 2048]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_13: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_36, add_7);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_39: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        amax_1: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_39, [-1], True)
        sub_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_39, amax_1);  convert_element_type_39 = amax_1 = None
        exp_1: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_40: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_3, torch.float16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_40, [1, 12, 2048, 2048]);  convert_element_type_40 = None
        view_39: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_7, [12, 2048, 2048]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_31: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_7, [2048, 768]);  mul_7 = None
        permute_16: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "f16[2048, 768]" = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
        view_32: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 2048, 768]);  mm_8 = None
        view_33: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_32, [1, 2048, -1, 64]);  view_32 = None
        permute_17: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_8: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_17, [1, 12, 2048, 64]);  permute_17 = None
        view_40: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_8, [12, 2048, 64]);  expand_8 = None
        bmm_3: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
        view_41: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 2048, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_7: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_42: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 2048, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_43: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_42, [2048, 768]);  view_42 = None
        permute_20: "f16[768, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "f16[2048, 768]" = torch.ops.aten.mm.default(view_43, permute_20);  view_43 = permute_20 = None
        view_44: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_9, [1, 2048, 768]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_14: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_28, view_44);  convert_element_type_28 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_45: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_2: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_14);  add_14 = None
        any_3: "b8[]" = torch.ops.aten.any.default(isinf_2);  isinf_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[]" = torch.ops.aten.where.self(any_3, full_default_8, full_default_7);  any_3 = full_default_8 = full_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_2: "f32[]" = torch.ops.aten.neg.default(where_4)
        clamp_min_2: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_45, neg_2);  convert_element_type_45 = neg_2 = None
        clamp_max_2: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_2, where_4);  clamp_min_2 = where_4 = None
        convert_element_type_46: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.float16);  clamp_max_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_47: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32)
        pow_4: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2);  convert_element_type_47 = None
        mean_3: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_8: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_46, rsqrt_3);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_48: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_8, torch.float16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_9: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg16_1, convert_element_type_48);  arg16_1 = convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_45: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_9, [2048, 768]);  mul_9 = None
        permute_21: "f16[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_10: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_10, [1, 2048, 3072]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_1: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_46);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_47: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_1, [2048, 3072]);  relu_1 = None
        permute_22: "f16[3072, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "f16[2048, 768]" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_11, [1, 2048, 768]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_16: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_46, view_48);  convert_element_type_46 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_53: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_3: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_16);  add_16 = None
        any_4: "b8[]" = torch.ops.aten.any.default(isinf_3);  isinf_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[]" = torch.ops.aten.where.self(any_4, full_default_10, full_default_9);  any_4 = full_default_10 = full_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_3: "f32[]" = torch.ops.aten.neg.default(where_5)
        clamp_min_3: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_53, neg_3);  convert_element_type_53 = neg_3 = None
        clamp_max_3: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_3, where_5);  clamp_min_3 = where_5 = None
        convert_element_type_54: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.float16);  clamp_max_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_55: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_54, torch.float32)
        pow_5: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_55, 2);  convert_element_type_55 = None
        mean_4: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_10: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_4);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_56: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.float16);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_11: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg19_1, convert_element_type_56);  arg19_1 = convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_49: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_11, [2048, 768])
        permute_23: "f16[768, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_12: "f16[2048, 768]" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_12, [1, 2048, 768]);  mm_12 = None
        view_51: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, 2048, -1, 64]);  view_50 = None
        permute_24: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_9: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_24, [1, 12, 2048, 64]);  permute_24 = None
        view_58: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_9, [12, 2048, 64]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_52: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_11, [2048, 768])
        permute_25: "f16[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_13: "f16[2048, 768]" = torch.ops.aten.mm.default(view_52, permute_25);  view_52 = permute_25 = None
        view_53: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_13, [1, 2048, 768]);  mm_13 = None
        view_54: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_53, [1, 2048, -1, 64]);  view_53 = None
        permute_26: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_29: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_10: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_29, [1, 12, 64, 2048]);  permute_29 = None
        view_59: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_10, [12, 64, 2048]);  expand_10 = None
        bmm_4: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 2048, 2048]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_18: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_60, add_7);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_65: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        amax_2: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_65, [-1], True)
        sub_3: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_65, amax_2);  convert_element_type_65 = amax_2 = None
        exp_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_66: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_4, torch.float16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_66, [1, 12, 2048, 2048]);  convert_element_type_66 = None
        view_63: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_11, [12, 2048, 2048]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_55: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_11, [2048, 768]);  mul_11 = None
        permute_27: "f16[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "f16[2048, 768]" = torch.ops.aten.mm.default(view_55, permute_27);  view_55 = permute_27 = None
        view_56: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_14, [1, 2048, 768]);  mm_14 = None
        view_57: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_56, [1, 2048, -1, 64]);  view_56 = None
        permute_28: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_12: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_28, [1, 12, 2048, 64]);  permute_28 = None
        view_64: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_12, [12, 2048, 64]);  expand_12 = None
        bmm_5: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
        view_65: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 2048, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_12: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_66: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_12, [1, 2048, -1]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_67: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_66, [2048, 768]);  view_66 = None
        permute_31: "f16[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "f16[2048, 768]" = torch.ops.aten.mm.default(view_67, permute_31);  view_67 = permute_31 = None
        view_68: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_15, [1, 2048, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_19: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_54, view_68);  convert_element_type_54 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_71: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_4: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_19);  add_19 = None
        any_5: "b8[]" = torch.ops.aten.any.default(isinf_4);  isinf_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[]" = torch.ops.aten.where.self(any_5, full_default_12, full_default_11);  any_5 = full_default_12 = full_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_4: "f32[]" = torch.ops.aten.neg.default(where_6)
        clamp_min_4: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_71, neg_4);  convert_element_type_71 = neg_4 = None
        clamp_max_4: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_4, where_6);  clamp_min_4 = where_6 = None
        convert_element_type_72: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.float16);  clamp_max_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_73: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_72, torch.float32)
        pow_6: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2);  convert_element_type_73 = None
        mean_5: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_12: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_5);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_74: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_12, torch.float16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_13: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg24_1, convert_element_type_74);  arg24_1 = convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_69: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_13, [2048, 768]);  mul_13 = None
        permute_32: "f16[768, 3072]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_16: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_69, permute_32);  view_69 = permute_32 = None
        view_70: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_16, [1, 2048, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_2: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_70);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_71: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_2, [2048, 3072]);  relu_2 = None
        permute_33: "f16[3072, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_17: "f16[2048, 768]" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_17, [1, 2048, 768]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_21: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_72, view_72);  convert_element_type_72 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_79: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_5: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_21);  add_21 = None
        any_6: "b8[]" = torch.ops.aten.any.default(isinf_5);  isinf_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[]" = torch.ops.aten.where.self(any_6, full_default_14, full_default_13);  any_6 = full_default_14 = full_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_5: "f32[]" = torch.ops.aten.neg.default(where_7)
        clamp_min_5: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_79, neg_5);  convert_element_type_79 = neg_5 = None
        clamp_max_5: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_5, where_7);  clamp_min_5 = where_7 = None
        convert_element_type_80: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_5, torch.float16);  clamp_max_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_81: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_80, torch.float32)
        pow_7: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_81, 2);  convert_element_type_81 = None
        mean_6: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_14: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_80, rsqrt_6);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_82: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_14, torch.float16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_15: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg27_1, convert_element_type_82);  arg27_1 = convert_element_type_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_73: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_15, [2048, 768])
        permute_34: "f16[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_18: "f16[2048, 768]" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_18, [1, 2048, 768]);  mm_18 = None
        view_75: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_74, [1, 2048, -1, 64]);  view_74 = None
        permute_35: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_13: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_35, [1, 12, 2048, 64]);  permute_35 = None
        view_82: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_13, [12, 2048, 64]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_76: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_15, [2048, 768])
        permute_36: "f16[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_19: "f16[2048, 768]" = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
        view_77: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_19, [1, 2048, 768]);  mm_19 = None
        view_78: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_77, [1, 2048, -1, 64]);  view_77 = None
        permute_37: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_40: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_14: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_40, [1, 12, 64, 2048]);  permute_40 = None
        view_83: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_14, [12, 64, 2048]);  expand_14 = None
        bmm_6: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 2048, 2048]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_23: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_84, add_7);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_91: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        amax_3: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_91, [-1], True)
        sub_4: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_91, amax_3);  convert_element_type_91 = amax_3 = None
        exp_3: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_92: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_5, torch.float16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_92, [1, 12, 2048, 2048]);  convert_element_type_92 = None
        view_87: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_15, [12, 2048, 2048]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_79: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_15, [2048, 768]);  mul_15 = None
        permute_38: "f16[768, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_20: "f16[2048, 768]" = torch.ops.aten.mm.default(view_79, permute_38);  view_79 = permute_38 = None
        view_80: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_20, [1, 2048, 768]);  mm_20 = None
        view_81: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_80, [1, 2048, -1, 64]);  view_80 = None
        permute_39: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_16: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_39, [1, 12, 2048, 64]);  permute_39 = None
        view_88: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_16, [12, 2048, 64]);  expand_16 = None
        bmm_7: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
        view_89: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 2048, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_17: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_90: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_17, [1, 2048, -1]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_91: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_90, [2048, 768]);  view_90 = None
        permute_42: "f16[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "f16[2048, 768]" = torch.ops.aten.mm.default(view_91, permute_42);  view_91 = permute_42 = None
        view_92: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_21, [1, 2048, 768]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_24: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_80, view_92);  convert_element_type_80 = view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_97: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_6: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_24);  add_24 = None
        any_7: "b8[]" = torch.ops.aten.any.default(isinf_6);  isinf_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[]" = torch.ops.aten.where.self(any_7, full_default_16, full_default_15);  any_7 = full_default_16 = full_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_6: "f32[]" = torch.ops.aten.neg.default(where_8)
        clamp_min_6: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_97, neg_6);  convert_element_type_97 = neg_6 = None
        clamp_max_6: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_6, where_8);  clamp_min_6 = where_8 = None
        convert_element_type_98: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_6, torch.float16);  clamp_max_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_99: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_98, torch.float32)
        pow_8: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_99, 2);  convert_element_type_99 = None
        mean_7: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_25: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_16: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_98, rsqrt_7);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_100: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_16, torch.float16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_17: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg32_1, convert_element_type_100);  arg32_1 = convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_93: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_17, [2048, 768]);  mul_17 = None
        permute_43: "f16[768, 3072]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_22: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_93, permute_43);  view_93 = permute_43 = None
        view_94: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_22, [1, 2048, 3072]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_3: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_94);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_95: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_3, [2048, 3072]);  relu_3 = None
        permute_44: "f16[3072, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_23: "f16[2048, 768]" = torch.ops.aten.mm.default(view_95, permute_44);  view_95 = permute_44 = None
        view_96: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_23, [1, 2048, 768]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_26: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_98, view_96);  convert_element_type_98 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_105: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_7: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_26);  add_26 = None
        any_8: "b8[]" = torch.ops.aten.any.default(isinf_7);  isinf_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[]" = torch.ops.aten.where.self(any_8, full_default_18, full_default_17);  any_8 = full_default_18 = full_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_7: "f32[]" = torch.ops.aten.neg.default(where_9)
        clamp_min_7: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_105, neg_7);  convert_element_type_105 = neg_7 = None
        clamp_max_7: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_7, where_9);  clamp_min_7 = where_9 = None
        convert_element_type_106: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_7, torch.float16);  clamp_max_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_107: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_106, torch.float32)
        pow_9: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 2);  convert_element_type_107 = None
        mean_8: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_18: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_106, rsqrt_8);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_108: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_18, torch.float16);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_19: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_108);  arg35_1 = convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_97: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_19, [2048, 768])
        permute_45: "f16[768, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_24: "f16[2048, 768]" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_24, [1, 2048, 768]);  mm_24 = None
        view_99: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_98, [1, 2048, -1, 64]);  view_98 = None
        permute_46: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_17: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_46, [1, 12, 2048, 64]);  permute_46 = None
        view_106: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_17, [12, 2048, 64]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_100: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_19, [2048, 768])
        permute_47: "f16[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_25: "f16[2048, 768]" = torch.ops.aten.mm.default(view_100, permute_47);  view_100 = permute_47 = None
        view_101: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_25, [1, 2048, 768]);  mm_25 = None
        view_102: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_101, [1, 2048, -1, 64]);  view_101 = None
        permute_48: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_51: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_18: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_51, [1, 12, 64, 2048]);  permute_51 = None
        view_107: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_18, [12, 64, 2048]);  expand_18 = None
        bmm_8: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 2048, 2048]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_28: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_108, add_7);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_117: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32);  add_28 = None
        amax_4: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_117, [-1], True)
        sub_5: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_117, amax_4);  convert_element_type_117 = amax_4 = None
        exp_4: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_118: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_6, torch.float16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_118, [1, 12, 2048, 2048]);  convert_element_type_118 = None
        view_111: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_19, [12, 2048, 2048]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_103: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_19, [2048, 768]);  mul_19 = None
        permute_49: "f16[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_26: "f16[2048, 768]" = torch.ops.aten.mm.default(view_103, permute_49);  view_103 = permute_49 = None
        view_104: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_26, [1, 2048, 768]);  mm_26 = None
        view_105: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_104, [1, 2048, -1, 64]);  view_104 = None
        permute_50: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_20: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_50, [1, 12, 2048, 64]);  permute_50 = None
        view_112: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 2048, 64]);  expand_20 = None
        bmm_9: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_111, view_112);  view_111 = view_112 = None
        view_113: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 2048, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_22: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_114: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_22, [1, 2048, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_115: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_114, [2048, 768]);  view_114 = None
        permute_53: "f16[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_27: "f16[2048, 768]" = torch.ops.aten.mm.default(view_115, permute_53);  view_115 = permute_53 = None
        view_116: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_27, [1, 2048, 768]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_29: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_106, view_116);  convert_element_type_106 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_123: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_8: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_29);  add_29 = None
        any_9: "b8[]" = torch.ops.aten.any.default(isinf_8);  isinf_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[]" = torch.ops.aten.where.self(any_9, full_default_20, full_default_19);  any_9 = full_default_20 = full_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_8: "f32[]" = torch.ops.aten.neg.default(where_10)
        clamp_min_8: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_123, neg_8);  convert_element_type_123 = neg_8 = None
        clamp_max_8: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_8, where_10);  clamp_min_8 = where_10 = None
        convert_element_type_124: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_8, torch.float16);  clamp_max_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_125: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_124, torch.float32)
        pow_10: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_125, 2);  convert_element_type_125 = None
        mean_9: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_20: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_124, rsqrt_9);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_126: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_20, torch.float16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_21: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg40_1, convert_element_type_126);  arg40_1 = convert_element_type_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_117: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_21, [2048, 768]);  mul_21 = None
        permute_54: "f16[768, 3072]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_28: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_117, permute_54);  view_117 = permute_54 = None
        view_118: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_28, [1, 2048, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_4: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_118);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_119: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_4, [2048, 3072]);  relu_4 = None
        permute_55: "f16[3072, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_29: "f16[2048, 768]" = torch.ops.aten.mm.default(view_119, permute_55);  view_119 = permute_55 = None
        view_120: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_29, [1, 2048, 768]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_31: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_124, view_120);  convert_element_type_124 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_131: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_9: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_31);  add_31 = None
        any_10: "b8[]" = torch.ops.aten.any.default(isinf_9);  isinf_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[]" = torch.ops.aten.where.self(any_10, full_default_22, full_default_21);  any_10 = full_default_22 = full_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_9: "f32[]" = torch.ops.aten.neg.default(where_11)
        clamp_min_9: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_131, neg_9);  convert_element_type_131 = neg_9 = None
        clamp_max_9: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_9, where_11);  clamp_min_9 = where_11 = None
        convert_element_type_132: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_9, torch.float16);  clamp_max_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_133: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_132, torch.float32)
        pow_11: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2);  convert_element_type_133 = None
        mean_10: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_22: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_132, rsqrt_10);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_134: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.float16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_23: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg43_1, convert_element_type_134);  arg43_1 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_121: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_23, [2048, 768])
        permute_56: "f16[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_30: "f16[2048, 768]" = torch.ops.aten.mm.default(view_121, permute_56);  view_121 = permute_56 = None
        view_122: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_30, [1, 2048, 768]);  mm_30 = None
        view_123: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_122, [1, 2048, -1, 64]);  view_122 = None
        permute_57: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_21: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_57, [1, 12, 2048, 64]);  permute_57 = None
        view_130: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_21, [12, 2048, 64]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_124: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_23, [2048, 768])
        permute_58: "f16[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_31: "f16[2048, 768]" = torch.ops.aten.mm.default(view_124, permute_58);  view_124 = permute_58 = None
        view_125: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_31, [1, 2048, 768]);  mm_31 = None
        view_126: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_125, [1, 2048, -1, 64]);  view_125 = None
        permute_59: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_62: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_22: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_62, [1, 12, 64, 2048]);  permute_62 = None
        view_131: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_22, [12, 64, 2048]);  expand_22 = None
        bmm_10: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 2048, 2048]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_33: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_132, add_7);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_143: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        amax_5: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_143, [-1], True)
        sub_6: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_143, amax_5);  convert_element_type_143 = amax_5 = None
        exp_5: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_144: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_7, torch.float16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_144, [1, 12, 2048, 2048]);  convert_element_type_144 = None
        view_135: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_23, [12, 2048, 2048]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_127: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_23, [2048, 768]);  mul_23 = None
        permute_60: "f16[768, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_32: "f16[2048, 768]" = torch.ops.aten.mm.default(view_127, permute_60);  view_127 = permute_60 = None
        view_128: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_32, [1, 2048, 768]);  mm_32 = None
        view_129: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_128, [1, 2048, -1, 64]);  view_128 = None
        permute_61: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_24: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_61, [1, 12, 2048, 64]);  permute_61 = None
        view_136: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_24, [12, 2048, 64]);  expand_24 = None
        bmm_11: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
        view_137: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 2048, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_27: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_138: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_27, [1, 2048, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_139: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_138, [2048, 768]);  view_138 = None
        permute_64: "f16[768, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_33: "f16[2048, 768]" = torch.ops.aten.mm.default(view_139, permute_64);  view_139 = permute_64 = None
        view_140: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_33, [1, 2048, 768]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_34: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_132, view_140);  convert_element_type_132 = view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_149: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_10: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_34);  add_34 = None
        any_11: "b8[]" = torch.ops.aten.any.default(isinf_10);  isinf_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[]" = torch.ops.aten.where.self(any_11, full_default_24, full_default_23);  any_11 = full_default_24 = full_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_10: "f32[]" = torch.ops.aten.neg.default(where_12)
        clamp_min_10: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_149, neg_10);  convert_element_type_149 = neg_10 = None
        clamp_max_10: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_10, where_12);  clamp_min_10 = where_12 = None
        convert_element_type_150: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_10, torch.float16);  clamp_max_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_151: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_150, torch.float32)
        pow_12: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_151, 2);  convert_element_type_151 = None
        mean_11: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_24: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_150, rsqrt_11);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_152: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_24, torch.float16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_25: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_152);  arg48_1 = convert_element_type_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_141: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_25, [2048, 768]);  mul_25 = None
        permute_65: "f16[768, 3072]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_34: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_141, permute_65);  view_141 = permute_65 = None
        view_142: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_34, [1, 2048, 3072]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_5: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_142);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_143: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_5, [2048, 3072]);  relu_5 = None
        permute_66: "f16[3072, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_35: "f16[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_66);  view_143 = permute_66 = None
        view_144: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_35, [1, 2048, 768]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_36: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_150, view_144);  convert_element_type_150 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_157: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_11: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_36);  add_36 = None
        any_12: "b8[]" = torch.ops.aten.any.default(isinf_11);  isinf_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_26: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[]" = torch.ops.aten.where.self(any_12, full_default_26, full_default_25);  any_12 = full_default_26 = full_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_11: "f32[]" = torch.ops.aten.neg.default(where_13)
        clamp_min_11: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_157, neg_11);  convert_element_type_157 = neg_11 = None
        clamp_max_11: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_11, where_13);  clamp_min_11 = where_13 = None
        convert_element_type_158: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_11, torch.float16);  clamp_max_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_159: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32)
        pow_13: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_159, 2);  convert_element_type_159 = None
        mean_12: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_37: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_26: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_158, rsqrt_12);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_160: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_26, torch.float16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_27: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg51_1, convert_element_type_160);  arg51_1 = convert_element_type_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_145: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_27, [2048, 768])
        permute_67: "f16[768, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_36: "f16[2048, 768]" = torch.ops.aten.mm.default(view_145, permute_67);  view_145 = permute_67 = None
        view_146: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_36, [1, 2048, 768]);  mm_36 = None
        view_147: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_146, [1, 2048, -1, 64]);  view_146 = None
        permute_68: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_25: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_68, [1, 12, 2048, 64]);  permute_68 = None
        view_154: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_25, [12, 2048, 64]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_148: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_27, [2048, 768])
        permute_69: "f16[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_37: "f16[2048, 768]" = torch.ops.aten.mm.default(view_148, permute_69);  view_148 = permute_69 = None
        view_149: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_37, [1, 2048, 768]);  mm_37 = None
        view_150: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_149, [1, 2048, -1, 64]);  view_149 = None
        permute_70: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_73: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_26: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_73, [1, 12, 64, 2048]);  permute_73 = None
        view_155: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_26, [12, 64, 2048]);  expand_26 = None
        bmm_12: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_154, view_155);  view_154 = view_155 = None
        view_156: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 2048, 2048]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_38: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_156, add_7);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_169: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        amax_6: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_169, [-1], True)
        sub_7: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_169, amax_6);  convert_element_type_169 = amax_6 = None
        exp_6: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_7: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_8: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_170: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_8, torch.float16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_27: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_170, [1, 12, 2048, 2048]);  convert_element_type_170 = None
        view_159: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_27, [12, 2048, 2048]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_151: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_27, [2048, 768]);  mul_27 = None
        permute_71: "f16[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_38: "f16[2048, 768]" = torch.ops.aten.mm.default(view_151, permute_71);  view_151 = permute_71 = None
        view_152: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_38, [1, 2048, 768]);  mm_38 = None
        view_153: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_152, [1, 2048, -1, 64]);  view_152 = None
        permute_72: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_28: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_72, [1, 12, 2048, 64]);  permute_72 = None
        view_160: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_28, [12, 2048, 64]);  expand_28 = None
        bmm_13: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_159, view_160);  view_159 = view_160 = None
        view_161: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 2048, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        clone_32: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_162: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_32, [1, 2048, -1]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_163: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_162, [2048, 768]);  view_162 = None
        permute_75: "f16[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_39: "f16[2048, 768]" = torch.ops.aten.mm.default(view_163, permute_75);  view_163 = permute_75 = None
        view_164: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_39, [1, 2048, 768]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_39: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_158, view_164);  convert_element_type_158 = view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_175: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_12: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_39);  add_39 = None
        any_13: "b8[]" = torch.ops.aten.any.default(isinf_12);  isinf_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_28: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f32[]" = torch.ops.aten.where.self(any_13, full_default_28, full_default_27);  any_13 = full_default_28 = full_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_12: "f32[]" = torch.ops.aten.neg.default(where_14)
        clamp_min_12: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_175, neg_12);  convert_element_type_175 = neg_12 = None
        clamp_max_12: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_12, where_14);  clamp_min_12 = where_14 = None
        convert_element_type_176: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_12, torch.float16);  clamp_max_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_177: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_176, torch.float32)
        pow_14: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_177, 2);  convert_element_type_177 = None
        mean_13: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_40: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_28: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_176, rsqrt_13);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_178: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_28, torch.float16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_29: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg56_1, convert_element_type_178);  arg56_1 = convert_element_type_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_165: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_29, [2048, 768]);  mul_29 = None
        permute_76: "f16[768, 3072]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_40: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_165, permute_76);  view_165 = permute_76 = None
        view_166: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_40, [1, 2048, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_6: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_166);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_167: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_6, [2048, 3072]);  relu_6 = None
        permute_77: "f16[3072, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_41: "f16[2048, 768]" = torch.ops.aten.mm.default(view_167, permute_77);  view_167 = permute_77 = None
        view_168: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_41, [1, 2048, 768]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_41: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_176, view_168);  convert_element_type_176 = view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_183: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_13: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_41);  add_41 = None
        any_14: "b8[]" = torch.ops.aten.any.default(isinf_13);  isinf_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_30: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f32[]" = torch.ops.aten.where.self(any_14, full_default_30, full_default_29);  any_14 = full_default_30 = full_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_13: "f32[]" = torch.ops.aten.neg.default(where_15)
        clamp_min_13: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_183, neg_13);  convert_element_type_183 = neg_13 = None
        clamp_max_13: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_13, where_15);  clamp_min_13 = where_15 = None
        convert_element_type_184: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_13, torch.float16);  clamp_max_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_185: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_184, torch.float32)
        pow_15: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_185, 2);  convert_element_type_185 = None
        mean_14: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_30: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_184, rsqrt_14);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_186: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_30, torch.float16);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_31: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg59_1, convert_element_type_186);  arg59_1 = convert_element_type_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_169: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_31, [2048, 768])
        permute_78: "f16[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_42: "f16[2048, 768]" = torch.ops.aten.mm.default(view_169, permute_78);  view_169 = permute_78 = None
        view_170: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_42, [1, 2048, 768]);  mm_42 = None
        view_171: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_170, [1, 2048, -1, 64]);  view_170 = None
        permute_79: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_29: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_79, [1, 12, 2048, 64]);  permute_79 = None
        view_178: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_29, [12, 2048, 64]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_172: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_31, [2048, 768])
        permute_80: "f16[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_43: "f16[2048, 768]" = torch.ops.aten.mm.default(view_172, permute_80);  view_172 = permute_80 = None
        view_173: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_43, [1, 2048, 768]);  mm_43 = None
        view_174: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_173, [1, 2048, -1, 64]);  view_173 = None
        permute_81: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_84: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        expand_30: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_84, [1, 12, 64, 2048]);  permute_84 = None
        view_179: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_30, [12, 64, 2048]);  expand_30 = None
        bmm_14: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_178, view_179);  view_178 = view_179 = None
        view_180: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 2048, 2048]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_43: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_180, add_7);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_195: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        amax_7: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_195, [-1], True)
        sub_8: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_195, amax_7);  convert_element_type_195 = amax_7 = None
        exp_7: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_8: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_9: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_196: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_9, torch.float16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_31: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_196, [1, 12, 2048, 2048]);  convert_element_type_196 = None
        view_183: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_31, [12, 2048, 2048]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_175: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_31, [2048, 768]);  mul_31 = None
        permute_82: "f16[768, 768]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_44: "f16[2048, 768]" = torch.ops.aten.mm.default(view_175, permute_82);  view_175 = permute_82 = None
        view_176: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_44, [1, 2048, 768]);  mm_44 = None
        view_177: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_176, [1, 2048, -1, 64]);  view_176 = None
        permute_83: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_32: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_83, [1, 12, 2048, 64]);  permute_83 = None
        view_184: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_32, [12, 2048, 64]);  expand_32 = None
        bmm_15: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_183, view_184);  view_183 = view_184 = None
        view_185: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 2048, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_37: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_186: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_37, [1, 2048, -1]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_187: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_186, [2048, 768]);  view_186 = None
        permute_86: "f16[768, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_45: "f16[2048, 768]" = torch.ops.aten.mm.default(view_187, permute_86);  view_187 = permute_86 = None
        view_188: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_45, [1, 2048, 768]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_44: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_184, view_188);  convert_element_type_184 = view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_201: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_14: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_44);  add_44 = None
        any_15: "b8[]" = torch.ops.aten.any.default(isinf_14);  isinf_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_32: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f32[]" = torch.ops.aten.where.self(any_15, full_default_32, full_default_31);  any_15 = full_default_32 = full_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_14: "f32[]" = torch.ops.aten.neg.default(where_16)
        clamp_min_14: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_201, neg_14);  convert_element_type_201 = neg_14 = None
        clamp_max_14: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_14, where_16);  clamp_min_14 = where_16 = None
        convert_element_type_202: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_14, torch.float16);  clamp_max_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_203: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_202, torch.float32)
        pow_16: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_203, 2);  convert_element_type_203 = None
        mean_15: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_32: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_202, rsqrt_15);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_204: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_32, torch.float16);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_33: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg64_1, convert_element_type_204);  arg64_1 = convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_189: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_33, [2048, 768]);  mul_33 = None
        permute_87: "f16[768, 3072]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_46: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_189, permute_87);  view_189 = permute_87 = None
        view_190: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_46, [1, 2048, 3072]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_7: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_190);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_191: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_7, [2048, 3072]);  relu_7 = None
        permute_88: "f16[3072, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_47: "f16[2048, 768]" = torch.ops.aten.mm.default(view_191, permute_88);  view_191 = permute_88 = None
        view_192: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_47, [1, 2048, 768]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_46: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_202, view_192);  convert_element_type_202 = view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_209: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_15: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_46);  add_46 = None
        any_16: "b8[]" = torch.ops.aten.any.default(isinf_15);  isinf_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f32[]" = torch.ops.aten.where.self(any_16, full_default_34, full_default_33);  any_16 = full_default_34 = full_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_15: "f32[]" = torch.ops.aten.neg.default(where_17)
        clamp_min_15: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_209, neg_15);  convert_element_type_209 = neg_15 = None
        clamp_max_15: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_15, where_17);  clamp_min_15 = where_17 = None
        convert_element_type_210: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_15, torch.float16);  clamp_max_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_211: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32)
        pow_17: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_211, 2);  convert_element_type_211 = None
        mean_16: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_47: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_34: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_210, rsqrt_16);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_212: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_34, torch.float16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_35: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg67_1, convert_element_type_212);  arg67_1 = convert_element_type_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_193: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_35, [2048, 768])
        permute_89: "f16[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_48: "f16[2048, 768]" = torch.ops.aten.mm.default(view_193, permute_89);  view_193 = permute_89 = None
        view_194: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_48, [1, 2048, 768]);  mm_48 = None
        view_195: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_194, [1, 2048, -1, 64]);  view_194 = None
        permute_90: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_33: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_90, [1, 12, 2048, 64]);  permute_90 = None
        view_202: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_33, [12, 2048, 64]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_196: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_35, [2048, 768])
        permute_91: "f16[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_49: "f16[2048, 768]" = torch.ops.aten.mm.default(view_196, permute_91);  view_196 = permute_91 = None
        view_197: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_49, [1, 2048, 768]);  mm_49 = None
        view_198: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_197, [1, 2048, -1, 64]);  view_197 = None
        permute_92: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_95: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        expand_34: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_95, [1, 12, 64, 2048]);  permute_95 = None
        view_203: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_34, [12, 64, 2048]);  expand_34 = None
        bmm_16: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_202, view_203);  view_202 = view_203 = None
        view_204: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 2048, 2048]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_48: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_204, add_7);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_221: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        amax_8: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_221, [-1], True)
        sub_9: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_221, amax_8);  convert_element_type_221 = amax_8 = None
        exp_8: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_9: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_10: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_222: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_10, torch.float16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_35: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_222, [1, 12, 2048, 2048]);  convert_element_type_222 = None
        view_207: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_35, [12, 2048, 2048]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_199: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_35, [2048, 768]);  mul_35 = None
        permute_93: "f16[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_50: "f16[2048, 768]" = torch.ops.aten.mm.default(view_199, permute_93);  view_199 = permute_93 = None
        view_200: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_50, [1, 2048, 768]);  mm_50 = None
        view_201: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_200, [1, 2048, -1, 64]);  view_200 = None
        permute_94: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_36: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_94, [1, 12, 2048, 64]);  permute_94 = None
        view_208: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_36, [12, 2048, 64]);  expand_36 = None
        bmm_17: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
        view_209: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 2048, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_42: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_210: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_42, [1, 2048, -1]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_211: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_210, [2048, 768]);  view_210 = None
        permute_97: "f16[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_51: "f16[2048, 768]" = torch.ops.aten.mm.default(view_211, permute_97);  view_211 = permute_97 = None
        view_212: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_51, [1, 2048, 768]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_49: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_210, view_212);  convert_element_type_210 = view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_227: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_16: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_49);  add_49 = None
        any_17: "b8[]" = torch.ops.aten.any.default(isinf_16);  isinf_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_36: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f32[]" = torch.ops.aten.where.self(any_17, full_default_36, full_default_35);  any_17 = full_default_36 = full_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_16: "f32[]" = torch.ops.aten.neg.default(where_18)
        clamp_min_16: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_227, neg_16);  convert_element_type_227 = neg_16 = None
        clamp_max_16: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_16, where_18);  clamp_min_16 = where_18 = None
        convert_element_type_228: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_16, torch.float16);  clamp_max_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_229: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_228, torch.float32)
        pow_18: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_229, 2);  convert_element_type_229 = None
        mean_17: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_50: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_36: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_228, rsqrt_17);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_230: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_36, torch.float16);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_37: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg72_1, convert_element_type_230);  arg72_1 = convert_element_type_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_213: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_37, [2048, 768]);  mul_37 = None
        permute_98: "f16[768, 3072]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_52: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_213, permute_98);  view_213 = permute_98 = None
        view_214: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_52, [1, 2048, 3072]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_8: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_214);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_215: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_8, [2048, 3072]);  relu_8 = None
        permute_99: "f16[3072, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_53: "f16[2048, 768]" = torch.ops.aten.mm.default(view_215, permute_99);  view_215 = permute_99 = None
        view_216: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_53, [1, 2048, 768]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_51: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_228, view_216);  convert_element_type_228 = view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_235: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_17: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_51);  add_51 = None
        any_18: "b8[]" = torch.ops.aten.any.default(isinf_17);  isinf_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_38: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[]" = torch.ops.aten.where.self(any_18, full_default_38, full_default_37);  any_18 = full_default_38 = full_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_17: "f32[]" = torch.ops.aten.neg.default(where_19)
        clamp_min_17: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_235, neg_17);  convert_element_type_235 = neg_17 = None
        clamp_max_17: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_17, where_19);  clamp_min_17 = where_19 = None
        convert_element_type_236: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_17, torch.float16);  clamp_max_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_237: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_236, torch.float32)
        pow_19: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_237, 2);  convert_element_type_237 = None
        mean_18: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_38: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_236, rsqrt_18);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_238: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_38, torch.float16);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_39: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_238);  arg75_1 = convert_element_type_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_217: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_39, [2048, 768])
        permute_100: "f16[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_54: "f16[2048, 768]" = torch.ops.aten.mm.default(view_217, permute_100);  view_217 = permute_100 = None
        view_218: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_54, [1, 2048, 768]);  mm_54 = None
        view_219: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_218, [1, 2048, -1, 64]);  view_218 = None
        permute_101: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_37: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_101, [1, 12, 2048, 64]);  permute_101 = None
        view_226: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_37, [12, 2048, 64]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_220: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_39, [2048, 768])
        permute_102: "f16[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_55: "f16[2048, 768]" = torch.ops.aten.mm.default(view_220, permute_102);  view_220 = permute_102 = None
        view_221: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_55, [1, 2048, 768]);  mm_55 = None
        view_222: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_221, [1, 2048, -1, 64]);  view_221 = None
        permute_103: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_106: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        expand_38: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_106, [1, 12, 64, 2048]);  permute_106 = None
        view_227: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_38, [12, 64, 2048]);  expand_38 = None
        bmm_18: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_226, view_227);  view_226 = view_227 = None
        view_228: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 2048, 2048]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_53: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_228, add_7);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_247: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        amax_9: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_247, [-1], True)
        sub_10: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_247, amax_9);  convert_element_type_247 = amax_9 = None
        exp_9: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_10: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_11: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_248: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_11, torch.float16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_39: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_248, [1, 12, 2048, 2048]);  convert_element_type_248 = None
        view_231: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_39, [12, 2048, 2048]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_223: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_39, [2048, 768]);  mul_39 = None
        permute_104: "f16[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_56: "f16[2048, 768]" = torch.ops.aten.mm.default(view_223, permute_104);  view_223 = permute_104 = None
        view_224: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_56, [1, 2048, 768]);  mm_56 = None
        view_225: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_224, [1, 2048, -1, 64]);  view_224 = None
        permute_105: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_40: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_105, [1, 12, 2048, 64]);  permute_105 = None
        view_232: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_40, [12, 2048, 64]);  expand_40 = None
        bmm_19: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
        view_233: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 2048, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        clone_47: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_234: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_47, [1, 2048, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_235: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_234, [2048, 768]);  view_234 = None
        permute_108: "f16[768, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_57: "f16[2048, 768]" = torch.ops.aten.mm.default(view_235, permute_108);  view_235 = permute_108 = None
        view_236: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_57, [1, 2048, 768]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_54: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_236, view_236);  convert_element_type_236 = view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_253: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_18: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_54);  add_54 = None
        any_19: "b8[]" = torch.ops.aten.any.default(isinf_18);  isinf_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_40: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[]" = torch.ops.aten.where.self(any_19, full_default_40, full_default_39);  any_19 = full_default_40 = full_default_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_18: "f32[]" = torch.ops.aten.neg.default(where_20)
        clamp_min_18: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_253, neg_18);  convert_element_type_253 = neg_18 = None
        clamp_max_18: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_18, where_20);  clamp_min_18 = where_20 = None
        convert_element_type_254: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_18, torch.float16);  clamp_max_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_255: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_254, torch.float32)
        pow_20: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_255, 2);  convert_element_type_255 = None
        mean_19: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_40: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_254, rsqrt_19);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_256: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_40, torch.float16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_41: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_256);  arg80_1 = convert_element_type_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_237: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_41, [2048, 768]);  mul_41 = None
        permute_109: "f16[768, 3072]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_58: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_237, permute_109);  view_237 = permute_109 = None
        view_238: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_58, [1, 2048, 3072]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_9: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_238);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_239: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_9, [2048, 3072]);  relu_9 = None
        permute_110: "f16[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_59: "f16[2048, 768]" = torch.ops.aten.mm.default(view_239, permute_110);  view_239 = permute_110 = None
        view_240: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_59, [1, 2048, 768]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_56: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_254, view_240);  convert_element_type_254 = view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_261: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_19: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_56);  add_56 = None
        any_20: "b8[]" = torch.ops.aten.any.default(isinf_19);  isinf_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_42: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "f32[]" = torch.ops.aten.where.self(any_20, full_default_42, full_default_41);  any_20 = full_default_42 = full_default_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_19: "f32[]" = torch.ops.aten.neg.default(where_21)
        clamp_min_19: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_261, neg_19);  convert_element_type_261 = neg_19 = None
        clamp_max_19: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_19, where_21);  clamp_min_19 = where_21 = None
        convert_element_type_262: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_19, torch.float16);  clamp_max_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_263: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_262, torch.float32)
        pow_21: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_263, 2);  convert_element_type_263 = None
        mean_20: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_57: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_42: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_262, rsqrt_20);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_264: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_42, torch.float16);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_43: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg83_1, convert_element_type_264);  arg83_1 = convert_element_type_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_241: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_43, [2048, 768])
        permute_111: "f16[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_60: "f16[2048, 768]" = torch.ops.aten.mm.default(view_241, permute_111);  view_241 = permute_111 = None
        view_242: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_60, [1, 2048, 768]);  mm_60 = None
        view_243: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_242, [1, 2048, -1, 64]);  view_242 = None
        permute_112: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_41: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_112, [1, 12, 2048, 64]);  permute_112 = None
        view_250: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_41, [12, 2048, 64]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_244: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_43, [2048, 768])
        permute_113: "f16[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_61: "f16[2048, 768]" = torch.ops.aten.mm.default(view_244, permute_113);  view_244 = permute_113 = None
        view_245: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_61, [1, 2048, 768]);  mm_61 = None
        view_246: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_245, [1, 2048, -1, 64]);  view_245 = None
        permute_114: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_117: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        expand_42: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_117, [1, 12, 64, 2048]);  permute_117 = None
        view_251: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_42, [12, 64, 2048]);  expand_42 = None
        bmm_20: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_250, view_251);  view_250 = view_251 = None
        view_252: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 2048, 2048]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_58: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_252, add_7);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_273: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None
        amax_10: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_273, [-1], True)
        sub_11: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_273, amax_10);  convert_element_type_273 = amax_10 = None
        exp_10: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_11: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_12: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_274: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_12, torch.float16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_43: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_274, [1, 12, 2048, 2048]);  convert_element_type_274 = None
        view_255: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_43, [12, 2048, 2048]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_247: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_43, [2048, 768]);  mul_43 = None
        permute_115: "f16[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_62: "f16[2048, 768]" = torch.ops.aten.mm.default(view_247, permute_115);  view_247 = permute_115 = None
        view_248: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_62, [1, 2048, 768]);  mm_62 = None
        view_249: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_248, [1, 2048, -1, 64]);  view_248 = None
        permute_116: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_44: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_116, [1, 12, 2048, 64]);  permute_116 = None
        view_256: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_44, [12, 2048, 64]);  expand_44 = None
        bmm_21: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_255, view_256);  view_255 = view_256 = None
        view_257: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 2048, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        clone_52: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_258: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_52, [1, 2048, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_259: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_258, [2048, 768]);  view_258 = None
        permute_119: "f16[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_63: "f16[2048, 768]" = torch.ops.aten.mm.default(view_259, permute_119);  view_259 = permute_119 = None
        view_260: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_63, [1, 2048, 768]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_59: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_262, view_260);  convert_element_type_262 = view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_279: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_20: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_59);  add_59 = None
        any_21: "b8[]" = torch.ops.aten.any.default(isinf_20);  isinf_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_44: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f32[]" = torch.ops.aten.where.self(any_21, full_default_44, full_default_43);  any_21 = full_default_44 = full_default_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_20: "f32[]" = torch.ops.aten.neg.default(where_22)
        clamp_min_20: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_279, neg_20);  convert_element_type_279 = neg_20 = None
        clamp_max_20: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_20, where_22);  clamp_min_20 = where_22 = None
        convert_element_type_280: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_20, torch.float16);  clamp_max_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_281: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_280, torch.float32)
        pow_22: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_281, 2);  convert_element_type_281 = None
        mean_21: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_60: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_44: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_280, rsqrt_21);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_282: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_45: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg88_1, convert_element_type_282);  arg88_1 = convert_element_type_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_261: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_45, [2048, 768]);  mul_45 = None
        permute_120: "f16[768, 3072]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_64: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_261, permute_120);  view_261 = permute_120 = None
        view_262: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_64, [1, 2048, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_10: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_262);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_263: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_10, [2048, 3072]);  relu_10 = None
        permute_121: "f16[3072, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_65: "f16[2048, 768]" = torch.ops.aten.mm.default(view_263, permute_121);  view_263 = permute_121 = None
        view_264: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_65, [1, 2048, 768]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_61: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_280, view_264);  convert_element_type_280 = view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_287: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_61, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_21: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_61);  add_61 = None
        any_22: "b8[]" = torch.ops.aten.any.default(isinf_21);  isinf_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_46: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f32[]" = torch.ops.aten.where.self(any_22, full_default_46, full_default_45);  any_22 = full_default_46 = full_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_21: "f32[]" = torch.ops.aten.neg.default(where_23)
        clamp_min_21: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_287, neg_21);  convert_element_type_287 = neg_21 = None
        clamp_max_21: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_21, where_23);  clamp_min_21 = where_23 = None
        convert_element_type_288: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_21, torch.float16);  clamp_max_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_289: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_288, torch.float32)
        pow_23: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_289, 2);  convert_element_type_289 = None
        mean_22: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_62: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_46: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_288, rsqrt_22);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_290: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_46, torch.float16);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_47: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg91_1, convert_element_type_290);  arg91_1 = convert_element_type_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_265: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_47, [2048, 768])
        permute_122: "f16[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_66: "f16[2048, 768]" = torch.ops.aten.mm.default(view_265, permute_122);  view_265 = permute_122 = None
        view_266: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_66, [1, 2048, 768]);  mm_66 = None
        view_267: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_266, [1, 2048, -1, 64]);  view_266 = None
        permute_123: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_45: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_123, [1, 12, 2048, 64]);  permute_123 = None
        view_274: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_45, [12, 2048, 64]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_268: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_47, [2048, 768])
        permute_124: "f16[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_67: "f16[2048, 768]" = torch.ops.aten.mm.default(view_268, permute_124);  view_268 = permute_124 = None
        view_269: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_67, [1, 2048, 768]);  mm_67 = None
        view_270: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_269, [1, 2048, -1, 64]);  view_269 = None
        permute_125: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_128: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        expand_46: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_128, [1, 12, 64, 2048]);  permute_128 = None
        view_275: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_46, [12, 64, 2048]);  expand_46 = None
        bmm_22: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_274, view_275);  view_274 = view_275 = None
        view_276: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 2048, 2048]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_63: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_276, add_7);  view_276 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_299: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        amax_11: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_299, [-1], True)
        sub_12: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_299, amax_11);  convert_element_type_299 = amax_11 = None
        exp_11: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_12: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_13: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_300: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_13, torch.float16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_47: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_300, [1, 12, 2048, 2048]);  convert_element_type_300 = None
        view_279: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_47, [12, 2048, 2048]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_271: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_47, [2048, 768]);  mul_47 = None
        permute_126: "f16[768, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_68: "f16[2048, 768]" = torch.ops.aten.mm.default(view_271, permute_126);  view_271 = permute_126 = None
        view_272: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_68, [1, 2048, 768]);  mm_68 = None
        view_273: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_272, [1, 2048, -1, 64]);  view_272 = None
        permute_127: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_48: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_127, [1, 12, 2048, 64]);  permute_127 = None
        view_280: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_48, [12, 2048, 64]);  expand_48 = None
        bmm_23: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_279, view_280);  view_279 = view_280 = None
        view_281: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 2048, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
        clone_57: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_282: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_57, [1, 2048, -1]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_283: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_282, [2048, 768]);  view_282 = None
        permute_130: "f16[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_69: "f16[2048, 768]" = torch.ops.aten.mm.default(view_283, permute_130);  view_283 = permute_130 = None
        view_284: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_69, [1, 2048, 768]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_64: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_288, view_284);  convert_element_type_288 = view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_305: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_22: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_64);  add_64 = None
        any_23: "b8[]" = torch.ops.aten.any.default(isinf_22);  isinf_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_48: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f32[]" = torch.ops.aten.where.self(any_23, full_default_48, full_default_47);  any_23 = full_default_48 = full_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_22: "f32[]" = torch.ops.aten.neg.default(where_24)
        clamp_min_22: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_305, neg_22);  convert_element_type_305 = neg_22 = None
        clamp_max_22: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_22, where_24);  clamp_min_22 = where_24 = None
        convert_element_type_306: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_22, torch.float16);  clamp_max_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_307: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_306, torch.float32)
        pow_24: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_307, 2);  convert_element_type_307 = None
        mean_23: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_65: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_48: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_306, rsqrt_23);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_308: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_48, torch.float16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_49: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg96_1, convert_element_type_308);  arg96_1 = convert_element_type_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_285: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_49, [2048, 768]);  mul_49 = None
        permute_131: "f16[768, 3072]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_70: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_285, permute_131);  view_285 = permute_131 = None
        view_286: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_70, [1, 2048, 3072]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_11: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_286);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_287: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_11, [2048, 3072]);  relu_11 = None
        permute_132: "f16[3072, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_71: "f16[2048, 768]" = torch.ops.aten.mm.default(view_287, permute_132);  view_287 = permute_132 = None
        view_288: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_71, [1, 2048, 768]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_66: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_306, view_288);  convert_element_type_306 = view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_313: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_23: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_66);  add_66 = None
        any_24: "b8[]" = torch.ops.aten.any.default(isinf_23);  isinf_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_50: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[]" = torch.ops.aten.where.self(any_24, full_default_50, full_default_49);  any_24 = full_default_50 = full_default_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_23: "f32[]" = torch.ops.aten.neg.default(where_25)
        clamp_min_23: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_313, neg_23);  convert_element_type_313 = neg_23 = None
        clamp_max_23: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_23, where_25);  clamp_min_23 = where_25 = None
        convert_element_type_314: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_23, torch.float16);  clamp_max_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_315: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_314, torch.float32)
        pow_25: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_315, 2);  convert_element_type_315 = None
        mean_24: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_67: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_50: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_314, rsqrt_24);  convert_element_type_314 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_316: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_50, torch.float16);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_51: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg99_1, convert_element_type_316);  arg99_1 = convert_element_type_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_313: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_145: "f16[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_77: "f16[2048, 768]" = torch.ops.aten.mm.default(view_313, permute_145);  view_313 = permute_145 = None
        view_314: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_77, [1, 2048, 768]);  mm_77 = None
        view_315: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_314, [1, 2048, -1, 64]);  view_314 = None
        permute_146: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_149: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_146, [0, 1, 3, 2]);  permute_146 = None
        expand_56: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_149, [1, 12, 64, 2048]);  permute_149 = None
        view_320: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_56, [12, 64, 2048]);  expand_56 = None
        bmm_26: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_319, view_320);  view_319 = view_320 = None
        view_321: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_26, [1, 12, 2048, 2048]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:300 in forward, code: position_bias = torch.zeros(
        full_default_59: "f16[1, 12, 2048, 2048]" = torch.ops.aten.full.default([1, 12, 2048, 2048], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_70: "i64[2048]" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_12: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_70, 0);  add_70 = None
        unsqueeze_13: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        unsqueeze_14: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 2048, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_14, 0);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_50: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(ge_1, [1, -1, 2048, 2048]);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_53: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand_50, full_default_53, full_default_54);  expand_50 = full_default_53 = full_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_80: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(full_default_59, where_27);  full_default_59 = where_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_81: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_321, add_80);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_347: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        amax_13: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_347, [-1], True)
        sub_15: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_347, amax_13);  convert_element_type_347 = amax_13 = None
        exp_13: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_348: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_17, torch.float16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_348, [1, 12, 2048, 2048]);  convert_element_type_348 = None
        view_324: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_57, [12, 2048, 2048]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_316: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_147: "f16[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_78: "f16[2048, 768]" = torch.ops.aten.mm.default(view_316, permute_147);  view_316 = permute_147 = None
        view_317: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_78, [1, 2048, 768]);  mm_78 = None
        view_318: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_317, [1, 2048, -1, 64]);  view_317 = None
        permute_148: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_58: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_148, [1, 12, 2048, 64]);  permute_148 = None
        view_325: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_58, [12, 2048, 64]);  expand_58 = None
        bmm_27: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
        view_326: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 12, 2048, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        clone_67: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_327: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_67, [1, 2048, -1]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_328: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_327, [2048, 768]);  view_327 = None
        permute_151: "f16[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_79: "f16[2048, 768]" = torch.ops.aten.mm.default(view_328, permute_151);  view_328 = permute_151 = None
        view_329: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_79, [1, 2048, 768]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_82: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_336, view_329);  convert_element_type_336 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_353: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_25: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_82);  add_82 = None
        any_26: "b8[]" = torch.ops.aten.any.default(isinf_25);  isinf_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_61: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "f32[]" = torch.ops.aten.where.self(any_26, full_default_61, full_default_60);  any_26 = full_default_61 = full_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_26: "f32[]" = torch.ops.aten.neg.default(where_30)
        clamp_min_25: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_353, neg_26);  convert_element_type_353 = neg_26 = None
        clamp_max_25: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_25, where_30);  clamp_min_25 = where_30 = None
        convert_element_type_354: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_25, torch.float16);  clamp_max_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_355: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_354, torch.float32)
        pow_28: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_355, 2);  convert_element_type_355 = None
        mean_27: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_83: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_57: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_354, rsqrt_27);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_356: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_57, torch.float16);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_58: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_356);  arg111_1 = convert_element_type_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_330: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_58, [2048, 768]);  mul_58 = None
        permute_152: "f16[768, 3072]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_80: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_330, permute_152);  view_330 = permute_152 = None
        view_331: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_80, [1, 2048, 3072]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_12: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_331);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_332: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_12, [2048, 3072]);  relu_12 = None
        permute_153: "f16[3072, 768]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_81: "f16[2048, 768]" = torch.ops.aten.mm.default(view_332, permute_153);  view_332 = permute_153 = None
        view_333: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_81, [1, 2048, 768]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_84: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_354, view_333);  convert_element_type_354 = view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_361: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_84, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_26: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_84);  add_84 = None
        any_27: "b8[]" = torch.ops.aten.any.default(isinf_26);  isinf_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_63: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "f32[]" = torch.ops.aten.where.self(any_27, full_default_63, full_default_62);  any_27 = full_default_63 = full_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_27: "f32[]" = torch.ops.aten.neg.default(where_31)
        clamp_min_26: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_361, neg_27);  convert_element_type_361 = neg_27 = None
        clamp_max_26: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_26, where_31);  clamp_min_26 = where_31 = None
        convert_element_type_362: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_26, torch.float16);  clamp_max_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_363: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_362, torch.float32)
        pow_29: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_363, 2);  convert_element_type_363 = None
        mean_28: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_59: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_362, rsqrt_28);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_364: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_59, torch.float16);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_60: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg114_1, convert_element_type_364);  arg114_1 = convert_element_type_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_334: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_60, [2048, 768])
        permute_154: "f16[768, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_82: "f16[2048, 768]" = torch.ops.aten.mm.default(view_334, permute_154);  view_334 = permute_154 = None
        view_335: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_82, [1, 2048, 768]);  mm_82 = None
        view_336: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_335, [1, 2048, -1, 64]);  view_335 = None
        permute_155: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_59: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_155, [1, 12, 2048, 64]);  permute_155 = None
        view_343: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_59, [12, 2048, 64]);  expand_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_337: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_60, [2048, 768])
        permute_156: "f16[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        mm_83: "f16[2048, 768]" = torch.ops.aten.mm.default(view_337, permute_156);  view_337 = permute_156 = None
        view_338: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_83, [1, 2048, 768]);  mm_83 = None
        view_339: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_338, [1, 2048, -1, 64]);  view_338 = None
        permute_157: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_160: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_157, [0, 1, 3, 2]);  permute_157 = None
        expand_60: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_160, [1, 12, 64, 2048]);  permute_160 = None
        view_344: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_60, [12, 64, 2048]);  expand_60 = None
        bmm_28: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_343, view_344);  view_343 = view_344 = None
        view_345: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_28, [1, 12, 2048, 2048]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_86: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_345, add_76);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_373: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32);  add_86 = None
        amax_14: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_373, [-1], True)
        sub_16: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_373, amax_14);  convert_element_type_373 = amax_14 = None
        exp_14: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_374: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_18, torch.float16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_374, [1, 12, 2048, 2048]);  convert_element_type_374 = None
        view_348: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_61, [12, 2048, 2048]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_340: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_60, [2048, 768]);  mul_60 = None
        permute_158: "f16[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_84: "f16[2048, 768]" = torch.ops.aten.mm.default(view_340, permute_158);  view_340 = permute_158 = None
        view_341: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_84, [1, 2048, 768]);  mm_84 = None
        view_342: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_341, [1, 2048, -1, 64]);  view_341 = None
        permute_159: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_62: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_159, [1, 12, 2048, 64]);  permute_159 = None
        view_349: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_62, [12, 2048, 64]);  expand_62 = None
        bmm_29: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_348, view_349);  view_348 = view_349 = None
        view_350: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 12, 2048, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        clone_72: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_351: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_72, [1, 2048, -1]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_352: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_351, [2048, 768]);  view_351 = None
        permute_162: "f16[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_85: "f16[2048, 768]" = torch.ops.aten.mm.default(view_352, permute_162);  view_352 = permute_162 = None
        view_353: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_85, [1, 2048, 768]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_87: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_362, view_353);  convert_element_type_362 = view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_379: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_27: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_87);  add_87 = None
        any_28: "b8[]" = torch.ops.aten.any.default(isinf_27);  isinf_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_65: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f32[]" = torch.ops.aten.where.self(any_28, full_default_65, full_default_64);  any_28 = full_default_65 = full_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_28: "f32[]" = torch.ops.aten.neg.default(where_32)
        clamp_min_27: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_379, neg_28);  convert_element_type_379 = neg_28 = None
        clamp_max_27: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_27, where_32);  clamp_min_27 = where_32 = None
        convert_element_type_380: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_27, torch.float16);  clamp_max_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_381: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_380, torch.float32)
        pow_30: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_381, 2);  convert_element_type_381 = None
        mean_29: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_88: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_61: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_380, rsqrt_29);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_382: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_61, torch.float16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_62: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg119_1, convert_element_type_382);  arg119_1 = convert_element_type_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_354: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_62, [2048, 768]);  mul_62 = None
        permute_163: "f16[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        mm_86: "f16[2048, 768]" = torch.ops.aten.mm.default(view_354, permute_163);  view_354 = permute_163 = None
        view_355: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_86, [1, 2048, 768]);  mm_86 = None
        view_356: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_355, [1, 2048, -1, 64]);  view_355 = None
        permute_164: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_63: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_164, [1, 12, 2048, 64]);  permute_164 = None
        view_363: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_63, [12, 2048, 64]);  expand_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_357: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_165: "f16[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_87: "f16[2048, 768]" = torch.ops.aten.mm.default(view_357, permute_165);  view_357 = permute_165 = None
        view_358: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_87, [1, 2048, 768]);  mm_87 = None
        view_359: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_358, [1, 2048, -1, 64]);  view_358 = None
        permute_166: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_169: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_166, [0, 1, 3, 2]);  permute_166 = None
        expand_64: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_169, [1, 12, 64, 2048]);  permute_169 = None
        view_364: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_64, [12, 64, 2048]);  expand_64 = None
        bmm_30: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_363, view_364);  view_363 = view_364 = None
        view_365: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_30, [1, 12, 2048, 2048]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_89: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_365, add_80);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_391: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        amax_15: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_391, [-1], True)
        sub_17: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_391, amax_15);  convert_element_type_391 = amax_15 = None
        exp_15: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_392: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_19, torch.float16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_392, [1, 12, 2048, 2048]);  convert_element_type_392 = None
        view_368: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_65, [12, 2048, 2048]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_360: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_167: "f16[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_88: "f16[2048, 768]" = torch.ops.aten.mm.default(view_360, permute_167);  view_360 = permute_167 = None
        view_361: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_88, [1, 2048, 768]);  mm_88 = None
        view_362: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_361, [1, 2048, -1, 64]);  view_361 = None
        permute_168: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_66: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_168, [1, 12, 2048, 64]);  permute_168 = None
        view_369: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_66, [12, 2048, 64]);  expand_66 = None
        bmm_31: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_368, view_369);  view_368 = view_369 = None
        view_370: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 12, 2048, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_170: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        clone_75: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_371: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_75, [1, 2048, -1]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_372: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_371, [2048, 768]);  view_371 = None
        permute_171: "f16[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_89: "f16[2048, 768]" = torch.ops.aten.mm.default(view_372, permute_171);  view_372 = permute_171 = None
        view_373: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_89, [1, 2048, 768]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_90: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_380, view_373);  convert_element_type_380 = view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_397: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_28: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_90);  add_90 = None
        any_29: "b8[]" = torch.ops.aten.any.default(isinf_28);  isinf_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_67: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33: "f32[]" = torch.ops.aten.where.self(any_29, full_default_67, full_default_66);  any_29 = full_default_67 = full_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_29: "f32[]" = torch.ops.aten.neg.default(where_33)
        clamp_min_28: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_397, neg_29);  convert_element_type_397 = neg_29 = None
        clamp_max_28: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_28, where_33);  clamp_min_28 = where_33 = None
        convert_element_type_398: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_28, torch.float16);  clamp_max_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_399: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_398, torch.float32)
        pow_31: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_399, 2);  convert_element_type_399 = None
        mean_30: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_91: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_63: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_398, rsqrt_30);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_400: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_63, torch.float16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_64: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg124_1, convert_element_type_400);  arg124_1 = convert_element_type_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_374: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_64, [2048, 768]);  mul_64 = None
        permute_172: "f16[768, 3072]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_90: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_374, permute_172);  view_374 = permute_172 = None
        view_375: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_90, [1, 2048, 3072]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_13: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_375);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_376: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_13, [2048, 3072]);  relu_13 = None
        permute_173: "f16[3072, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_91: "f16[2048, 768]" = torch.ops.aten.mm.default(view_376, permute_173);  view_376 = permute_173 = None
        view_377: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_91, [1, 2048, 768]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_92: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_398, view_377);  convert_element_type_398 = view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_405: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_29: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_92);  add_92 = None
        any_30: "b8[]" = torch.ops.aten.any.default(isinf_29);  isinf_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_69: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "f32[]" = torch.ops.aten.where.self(any_30, full_default_69, full_default_68);  any_30 = full_default_69 = full_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_30: "f32[]" = torch.ops.aten.neg.default(where_34)
        clamp_min_29: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_405, neg_30);  convert_element_type_405 = neg_30 = None
        clamp_max_29: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_29, where_34);  clamp_min_29 = where_34 = None
        convert_element_type_406: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_29, torch.float16);  clamp_max_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_407: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_406, torch.float32)
        pow_32: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_407, 2);  convert_element_type_407 = None
        mean_31: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_65: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_406, rsqrt_31);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_408: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_65, torch.float16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_66: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg127_1, convert_element_type_408);  arg127_1 = convert_element_type_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_378: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_66, [2048, 768])
        permute_174: "f16[768, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_92: "f16[2048, 768]" = torch.ops.aten.mm.default(view_378, permute_174);  view_378 = permute_174 = None
        view_379: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_92, [1, 2048, 768]);  mm_92 = None
        view_380: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_379, [1, 2048, -1, 64]);  view_379 = None
        permute_175: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_67: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_175, [1, 12, 2048, 64]);  permute_175 = None
        view_387: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_67, [12, 2048, 64]);  expand_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_381: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_66, [2048, 768])
        permute_176: "f16[768, 768]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_93: "f16[2048, 768]" = torch.ops.aten.mm.default(view_381, permute_176);  view_381 = permute_176 = None
        view_382: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_93, [1, 2048, 768]);  mm_93 = None
        view_383: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_382, [1, 2048, -1, 64]);  view_382 = None
        permute_177: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_180: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_177, [0, 1, 3, 2]);  permute_177 = None
        expand_68: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_180, [1, 12, 64, 2048]);  permute_180 = None
        view_388: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_68, [12, 64, 2048]);  expand_68 = None
        bmm_32: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_387, view_388);  view_387 = view_388 = None
        view_389: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_32, [1, 12, 2048, 2048]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_94: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_389, add_76);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_417: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32);  add_94 = None
        amax_16: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_417, [-1], True)
        sub_18: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_417, amax_16);  convert_element_type_417 = amax_16 = None
        exp_16: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_418: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_20, torch.float16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_418, [1, 12, 2048, 2048]);  convert_element_type_418 = None
        view_392: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_69, [12, 2048, 2048]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_384: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_66, [2048, 768]);  mul_66 = None
        permute_178: "f16[768, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_94: "f16[2048, 768]" = torch.ops.aten.mm.default(view_384, permute_178);  view_384 = permute_178 = None
        view_385: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_94, [1, 2048, 768]);  mm_94 = None
        view_386: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_385, [1, 2048, -1, 64]);  view_385 = None
        permute_179: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_70: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_179, [1, 12, 2048, 64]);  permute_179 = None
        view_393: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_70, [12, 2048, 64]);  expand_70 = None
        bmm_33: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_392, view_393);  view_392 = view_393 = None
        view_394: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 12, 2048, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_181: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
        clone_80: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_395: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_80, [1, 2048, -1]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_396: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_395, [2048, 768]);  view_395 = None
        permute_182: "f16[768, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_95: "f16[2048, 768]" = torch.ops.aten.mm.default(view_396, permute_182);  view_396 = permute_182 = None
        view_397: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_95, [1, 2048, 768]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_95: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_406, view_397);  convert_element_type_406 = view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_423: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_30: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_95);  add_95 = None
        any_31: "b8[]" = torch.ops.aten.any.default(isinf_30);  isinf_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_71: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35: "f32[]" = torch.ops.aten.where.self(any_31, full_default_71, full_default_70);  any_31 = full_default_71 = full_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_31: "f32[]" = torch.ops.aten.neg.default(where_35)
        clamp_min_30: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_423, neg_31);  convert_element_type_423 = neg_31 = None
        clamp_max_30: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_30, where_35);  clamp_min_30 = where_35 = None
        convert_element_type_424: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_30, torch.float16);  clamp_max_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_425: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_424, torch.float32)
        pow_33: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_425, 2);  convert_element_type_425 = None
        mean_32: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_96: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_67: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_424, rsqrt_32);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_426: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_67, torch.float16);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_68: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg132_1, convert_element_type_426);  arg132_1 = convert_element_type_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_398: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_68, [2048, 768]);  mul_68 = None
        permute_183: "f16[768, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_96: "f16[2048, 768]" = torch.ops.aten.mm.default(view_398, permute_183);  view_398 = permute_183 = None
        view_399: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_96, [1, 2048, 768]);  mm_96 = None
        view_400: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_399, [1, 2048, -1, 64]);  view_399 = None
        permute_184: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_71: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_184, [1, 12, 2048, 64]);  permute_184 = None
        view_407: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_71, [12, 2048, 64]);  expand_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_401: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_185: "f16[768, 768]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        mm_97: "f16[2048, 768]" = torch.ops.aten.mm.default(view_401, permute_185);  view_401 = permute_185 = None
        view_402: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_97, [1, 2048, 768]);  mm_97 = None
        view_403: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_402, [1, 2048, -1, 64]);  view_402 = None
        permute_186: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_189: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        expand_72: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_189, [1, 12, 64, 2048]);  permute_189 = None
        view_408: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_72, [12, 64, 2048]);  expand_72 = None
        bmm_34: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_407, view_408);  view_407 = view_408 = None
        view_409: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_34, [1, 12, 2048, 2048]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_97: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_409, add_80);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_435: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None
        amax_17: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_435, [-1], True)
        sub_19: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_435, amax_17);  convert_element_type_435 = amax_17 = None
        exp_17: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_436: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_21, torch.float16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_436, [1, 12, 2048, 2048]);  convert_element_type_436 = None
        view_412: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_73, [12, 2048, 2048]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_404: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_187: "f16[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_98: "f16[2048, 768]" = torch.ops.aten.mm.default(view_404, permute_187);  view_404 = permute_187 = None
        view_405: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_98, [1, 2048, 768]);  mm_98 = None
        view_406: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_405, [1, 2048, -1, 64]);  view_405 = None
        permute_188: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_74: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_188, [1, 12, 2048, 64]);  permute_188 = None
        view_413: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_74, [12, 2048, 64]);  expand_74 = None
        bmm_35: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_412, view_413);  view_412 = view_413 = None
        view_414: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 12, 2048, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_190: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        clone_83: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_415: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_83, [1, 2048, -1]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_416: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_415, [2048, 768]);  view_415 = None
        permute_191: "f16[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_99: "f16[2048, 768]" = torch.ops.aten.mm.default(view_416, permute_191);  view_416 = permute_191 = None
        view_417: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_99, [1, 2048, 768]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_98: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_424, view_417);  convert_element_type_424 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_441: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_98, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_31: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_98);  add_98 = None
        any_32: "b8[]" = torch.ops.aten.any.default(isinf_31);  isinf_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_73: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: "f32[]" = torch.ops.aten.where.self(any_32, full_default_73, full_default_72);  any_32 = full_default_73 = full_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_32: "f32[]" = torch.ops.aten.neg.default(where_36)
        clamp_min_31: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_441, neg_32);  convert_element_type_441 = neg_32 = None
        clamp_max_31: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_31, where_36);  clamp_min_31 = where_36 = None
        convert_element_type_442: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_31, torch.float16);  clamp_max_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_443: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_442, torch.float32)
        pow_34: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_443, 2);  convert_element_type_443 = None
        mean_33: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_99: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_69: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_442, rsqrt_33);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_444: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_70: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg137_1, convert_element_type_444);  arg137_1 = convert_element_type_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_418: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_70, [2048, 768]);  mul_70 = None
        permute_192: "f16[768, 3072]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_100: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_418, permute_192);  view_418 = permute_192 = None
        view_419: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_100, [1, 2048, 3072]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_14: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_419);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_420: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_14, [2048, 3072]);  relu_14 = None
        permute_193: "f16[3072, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_101: "f16[2048, 768]" = torch.ops.aten.mm.default(view_420, permute_193);  view_420 = permute_193 = None
        view_421: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_101, [1, 2048, 768]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_100: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_442, view_421);  convert_element_type_442 = view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_449: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_32: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_100);  add_100 = None
        any_33: "b8[]" = torch.ops.aten.any.default(isinf_32);  isinf_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_75: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37: "f32[]" = torch.ops.aten.where.self(any_33, full_default_75, full_default_74);  any_33 = full_default_75 = full_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_33: "f32[]" = torch.ops.aten.neg.default(where_37)
        clamp_min_32: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_449, neg_33);  convert_element_type_449 = neg_33 = None
        clamp_max_32: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_32, where_37);  clamp_min_32 = where_37 = None
        convert_element_type_450: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_32, torch.float16);  clamp_max_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_451: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_450, torch.float32)
        pow_35: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_451, 2);  convert_element_type_451 = None
        mean_34: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_101: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_71: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_450, rsqrt_34);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_452: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_71, torch.float16);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_72: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg140_1, convert_element_type_452);  arg140_1 = convert_element_type_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_422: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_72, [2048, 768])
        permute_194: "f16[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_102: "f16[2048, 768]" = torch.ops.aten.mm.default(view_422, permute_194);  view_422 = permute_194 = None
        view_423: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_102, [1, 2048, 768]);  mm_102 = None
        view_424: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_423, [1, 2048, -1, 64]);  view_423 = None
        permute_195: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_75: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_195, [1, 12, 2048, 64]);  permute_195 = None
        view_431: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_75, [12, 2048, 64]);  expand_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_425: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_72, [2048, 768])
        permute_196: "f16[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_103: "f16[2048, 768]" = torch.ops.aten.mm.default(view_425, permute_196);  view_425 = permute_196 = None
        view_426: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_103, [1, 2048, 768]);  mm_103 = None
        view_427: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_426, [1, 2048, -1, 64]);  view_426 = None
        permute_197: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_200: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_197, [0, 1, 3, 2]);  permute_197 = None
        expand_76: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_200, [1, 12, 64, 2048]);  permute_200 = None
        view_432: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_76, [12, 64, 2048]);  expand_76 = None
        bmm_36: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_431, view_432);  view_431 = view_432 = None
        view_433: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_36, [1, 12, 2048, 2048]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_102: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_433, add_76);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_461: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        amax_18: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_461, [-1], True)
        sub_20: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_461, amax_18);  convert_element_type_461 = amax_18 = None
        exp_18: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_19: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_462: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_22, torch.float16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_77: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_462, [1, 12, 2048, 2048]);  convert_element_type_462 = None
        view_436: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_77, [12, 2048, 2048]);  expand_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_428: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_72, [2048, 768]);  mul_72 = None
        permute_198: "f16[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_104: "f16[2048, 768]" = torch.ops.aten.mm.default(view_428, permute_198);  view_428 = permute_198 = None
        view_429: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_104, [1, 2048, 768]);  mm_104 = None
        view_430: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_429, [1, 2048, -1, 64]);  view_429 = None
        permute_199: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_78: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_199, [1, 12, 2048, 64]);  permute_199 = None
        view_437: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_78, [12, 2048, 64]);  expand_78 = None
        bmm_37: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_436, view_437);  view_436 = view_437 = None
        view_438: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 12, 2048, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_201: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None
        clone_88: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_439: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_88, [1, 2048, -1]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_440: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_439, [2048, 768]);  view_439 = None
        permute_202: "f16[768, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        mm_105: "f16[2048, 768]" = torch.ops.aten.mm.default(view_440, permute_202);  view_440 = permute_202 = None
        view_441: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_105, [1, 2048, 768]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_103: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_450, view_441);  convert_element_type_450 = view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_467: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_103, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_33: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_103);  add_103 = None
        any_34: "b8[]" = torch.ops.aten.any.default(isinf_33);  isinf_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_77: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38: "f32[]" = torch.ops.aten.where.self(any_34, full_default_77, full_default_76);  any_34 = full_default_77 = full_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_34: "f32[]" = torch.ops.aten.neg.default(where_38)
        clamp_min_33: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_467, neg_34);  convert_element_type_467 = neg_34 = None
        clamp_max_33: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_33, where_38);  clamp_min_33 = where_38 = None
        convert_element_type_468: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_33, torch.float16);  clamp_max_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_469: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_468, torch.float32)
        pow_36: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_469, 2);  convert_element_type_469 = None
        mean_35: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_104: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_73: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_468, rsqrt_35);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_470: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_73, torch.float16);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_74: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg145_1, convert_element_type_470);  arg145_1 = convert_element_type_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_442: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_74, [2048, 768]);  mul_74 = None
        permute_203: "f16[768, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_106: "f16[2048, 768]" = torch.ops.aten.mm.default(view_442, permute_203);  view_442 = permute_203 = None
        view_443: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_106, [1, 2048, 768]);  mm_106 = None
        view_444: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_443, [1, 2048, -1, 64]);  view_443 = None
        permute_204: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_79: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_204, [1, 12, 2048, 64]);  permute_204 = None
        view_451: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_79, [12, 2048, 64]);  expand_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_445: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_205: "f16[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_107: "f16[2048, 768]" = torch.ops.aten.mm.default(view_445, permute_205);  view_445 = permute_205 = None
        view_446: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_107, [1, 2048, 768]);  mm_107 = None
        view_447: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_446, [1, 2048, -1, 64]);  view_446 = None
        permute_206: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_209: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        expand_80: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_209, [1, 12, 64, 2048]);  permute_209 = None
        view_452: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_80, [12, 64, 2048]);  expand_80 = None
        bmm_38: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_451, view_452);  view_451 = view_452 = None
        view_453: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_38, [1, 12, 2048, 2048]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_105: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_453, add_80);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_479: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        amax_19: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_479, [-1], True)
        sub_21: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_479, amax_19);  convert_element_type_479 = amax_19 = None
        exp_19: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_20: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_480: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_23, torch.float16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_81: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_480, [1, 12, 2048, 2048]);  convert_element_type_480 = None
        view_456: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_81, [12, 2048, 2048]);  expand_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_448: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_207: "f16[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        mm_108: "f16[2048, 768]" = torch.ops.aten.mm.default(view_448, permute_207);  view_448 = permute_207 = None
        view_449: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_108, [1, 2048, 768]);  mm_108 = None
        view_450: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_449, [1, 2048, -1, 64]);  view_449 = None
        permute_208: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_82: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_208, [1, 12, 2048, 64]);  permute_208 = None
        view_457: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_82, [12, 2048, 64]);  expand_82 = None
        bmm_39: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_456, view_457);  view_456 = view_457 = None
        view_458: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 12, 2048, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_91: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_459: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_91, [1, 2048, -1]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_460: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_459, [2048, 768]);  view_459 = None
        permute_211: "f16[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_109: "f16[2048, 768]" = torch.ops.aten.mm.default(view_460, permute_211);  view_460 = permute_211 = None
        view_461: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_109, [1, 2048, 768]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_106: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_468, view_461);  convert_element_type_468 = view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_485: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_106, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_34: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_106);  add_106 = None
        any_35: "b8[]" = torch.ops.aten.any.default(isinf_34);  isinf_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_79: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39: "f32[]" = torch.ops.aten.where.self(any_35, full_default_79, full_default_78);  any_35 = full_default_79 = full_default_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_35: "f32[]" = torch.ops.aten.neg.default(where_39)
        clamp_min_34: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_485, neg_35);  convert_element_type_485 = neg_35 = None
        clamp_max_34: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_34, where_39);  clamp_min_34 = where_39 = None
        convert_element_type_486: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_34, torch.float16);  clamp_max_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_487: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_486, torch.float32)
        pow_37: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 2);  convert_element_type_487 = None
        mean_36: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_107: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_75: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_486, rsqrt_36);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_488: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_75, torch.float16);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_76: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg150_1, convert_element_type_488);  arg150_1 = convert_element_type_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_462: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_76, [2048, 768]);  mul_76 = None
        permute_212: "f16[768, 3072]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_110: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_462, permute_212);  view_462 = permute_212 = None
        view_463: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_110, [1, 2048, 3072]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_15: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_463);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_464: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_15, [2048, 3072]);  relu_15 = None
        permute_213: "f16[3072, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_111: "f16[2048, 768]" = torch.ops.aten.mm.default(view_464, permute_213);  view_464 = permute_213 = None
        view_465: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_111, [1, 2048, 768]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_108: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_486, view_465);  convert_element_type_486 = view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_493: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_108, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_35: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_108);  add_108 = None
        any_36: "b8[]" = torch.ops.aten.any.default(isinf_35);  isinf_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_81: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "f32[]" = torch.ops.aten.where.self(any_36, full_default_81, full_default_80);  any_36 = full_default_81 = full_default_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_36: "f32[]" = torch.ops.aten.neg.default(where_40)
        clamp_min_35: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_493, neg_36);  convert_element_type_493 = neg_36 = None
        clamp_max_35: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_35, where_40);  clamp_min_35 = where_40 = None
        convert_element_type_494: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_35, torch.float16);  clamp_max_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_495: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_494, torch.float32)
        pow_38: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_495, 2);  convert_element_type_495 = None
        mean_37: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_109: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_77: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_494, rsqrt_37);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_496: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_77, torch.float16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_78: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg153_1, convert_element_type_496);  arg153_1 = convert_element_type_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_466: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_78, [2048, 768])
        permute_214: "f16[768, 768]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_112: "f16[2048, 768]" = torch.ops.aten.mm.default(view_466, permute_214);  view_466 = permute_214 = None
        view_467: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_112, [1, 2048, 768]);  mm_112 = None
        view_468: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_467, [1, 2048, -1, 64]);  view_467 = None
        permute_215: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_83: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_215, [1, 12, 2048, 64]);  permute_215 = None
        view_475: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_83, [12, 2048, 64]);  expand_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_469: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_78, [2048, 768])
        permute_216: "f16[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_113: "f16[2048, 768]" = torch.ops.aten.mm.default(view_469, permute_216);  view_469 = permute_216 = None
        view_470: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_113, [1, 2048, 768]);  mm_113 = None
        view_471: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_470, [1, 2048, -1, 64]);  view_470 = None
        permute_217: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_220: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_217, [0, 1, 3, 2]);  permute_217 = None
        expand_84: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_220, [1, 12, 64, 2048]);  permute_220 = None
        view_476: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_84, [12, 64, 2048]);  expand_84 = None
        bmm_40: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_475, view_476);  view_475 = view_476 = None
        view_477: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_40, [1, 12, 2048, 2048]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_110: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_477, add_76);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_505: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_110, torch.float32);  add_110 = None
        amax_20: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_505, [-1], True)
        sub_22: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_505, amax_20);  convert_element_type_505 = amax_20 = None
        exp_20: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_21: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_506: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_24, torch.float16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_85: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_506, [1, 12, 2048, 2048]);  convert_element_type_506 = None
        view_480: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_85, [12, 2048, 2048]);  expand_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_472: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_78, [2048, 768]);  mul_78 = None
        permute_218: "f16[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_114: "f16[2048, 768]" = torch.ops.aten.mm.default(view_472, permute_218);  view_472 = permute_218 = None
        view_473: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_114, [1, 2048, 768]);  mm_114 = None
        view_474: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_473, [1, 2048, -1, 64]);  view_473 = None
        permute_219: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_86: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_219, [1, 12, 2048, 64]);  permute_219 = None
        view_481: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_86, [12, 2048, 64]);  expand_86 = None
        bmm_41: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_480, view_481);  view_480 = view_481 = None
        view_482: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 12, 2048, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        clone_96: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_483: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_96, [1, 2048, -1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_484: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_483, [2048, 768]);  view_483 = None
        permute_222: "f16[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        mm_115: "f16[2048, 768]" = torch.ops.aten.mm.default(view_484, permute_222);  view_484 = permute_222 = None
        view_485: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_115, [1, 2048, 768]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_111: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_494, view_485);  convert_element_type_494 = view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_511: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_36: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_111);  add_111 = None
        any_37: "b8[]" = torch.ops.aten.any.default(isinf_36);  isinf_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_83: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_41: "f32[]" = torch.ops.aten.where.self(any_37, full_default_83, full_default_82);  any_37 = full_default_83 = full_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_37: "f32[]" = torch.ops.aten.neg.default(where_41)
        clamp_min_36: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_511, neg_37);  convert_element_type_511 = neg_37 = None
        clamp_max_36: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_36, where_41);  clamp_min_36 = where_41 = None
        convert_element_type_512: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_36, torch.float16);  clamp_max_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_513: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_512, torch.float32)
        pow_39: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_513, 2);  convert_element_type_513 = None
        mean_38: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_79: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_512, rsqrt_38);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_514: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_79, torch.float16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_80: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg158_1, convert_element_type_514);  arg158_1 = convert_element_type_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_486: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_80, [2048, 768]);  mul_80 = None
        permute_223: "f16[768, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_116: "f16[2048, 768]" = torch.ops.aten.mm.default(view_486, permute_223);  view_486 = permute_223 = None
        view_487: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_116, [1, 2048, 768]);  mm_116 = None
        view_488: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_487, [1, 2048, -1, 64]);  view_487 = None
        permute_224: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_87: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_224, [1, 12, 2048, 64]);  permute_224 = None
        view_495: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_87, [12, 2048, 64]);  expand_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_489: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_225: "f16[768, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_117: "f16[2048, 768]" = torch.ops.aten.mm.default(view_489, permute_225);  view_489 = permute_225 = None
        view_490: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_117, [1, 2048, 768]);  mm_117 = None
        view_491: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_490, [1, 2048, -1, 64]);  view_490 = None
        permute_226: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_229: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_226, [0, 1, 3, 2]);  permute_226 = None
        expand_88: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_229, [1, 12, 64, 2048]);  permute_229 = None
        view_496: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_88, [12, 64, 2048]);  expand_88 = None
        bmm_42: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_495, view_496);  view_495 = view_496 = None
        view_497: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_42, [1, 12, 2048, 2048]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_113: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_497, add_80);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_523: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        amax_21: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_523, [-1], True)
        sub_23: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_523, amax_21);  convert_element_type_523 = amax_21 = None
        exp_21: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_22: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_524: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_25, torch.float16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_89: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_524, [1, 12, 2048, 2048]);  convert_element_type_524 = None
        view_500: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_89, [12, 2048, 2048]);  expand_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_492: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_227: "f16[768, 768]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_118: "f16[2048, 768]" = torch.ops.aten.mm.default(view_492, permute_227);  view_492 = permute_227 = None
        view_493: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_118, [1, 2048, 768]);  mm_118 = None
        view_494: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_493, [1, 2048, -1, 64]);  view_493 = None
        permute_228: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_90: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_228, [1, 12, 2048, 64]);  permute_228 = None
        view_501: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_90, [12, 2048, 64]);  expand_90 = None
        bmm_43: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_500, view_501);  view_500 = view_501 = None
        view_502: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 12, 2048, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
        clone_99: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_503: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_99, [1, 2048, -1]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_504: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_503, [2048, 768]);  view_503 = None
        permute_231: "f16[768, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_119: "f16[2048, 768]" = torch.ops.aten.mm.default(view_504, permute_231);  view_504 = permute_231 = None
        view_505: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_119, [1, 2048, 768]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_114: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_512, view_505);  convert_element_type_512 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_529: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_37: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_114);  add_114 = None
        any_38: "b8[]" = torch.ops.aten.any.default(isinf_37);  isinf_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_85: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42: "f32[]" = torch.ops.aten.where.self(any_38, full_default_85, full_default_84);  any_38 = full_default_85 = full_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_38: "f32[]" = torch.ops.aten.neg.default(where_42)
        clamp_min_37: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_529, neg_38);  convert_element_type_529 = neg_38 = None
        clamp_max_37: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_37, where_42);  clamp_min_37 = where_42 = None
        convert_element_type_530: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_37, torch.float16);  clamp_max_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_531: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_530, torch.float32)
        pow_40: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_531, 2);  convert_element_type_531 = None
        mean_39: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_115: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_81: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_530, rsqrt_39);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_532: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_81, torch.float16);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_82: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg163_1, convert_element_type_532);  arg163_1 = convert_element_type_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_506: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_82, [2048, 768]);  mul_82 = None
        permute_232: "f16[768, 3072]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_120: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_506, permute_232);  view_506 = permute_232 = None
        view_507: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_120, [1, 2048, 3072]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_16: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_507);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_508: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_16, [2048, 3072]);  relu_16 = None
        permute_233: "f16[3072, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_121: "f16[2048, 768]" = torch.ops.aten.mm.default(view_508, permute_233);  view_508 = permute_233 = None
        view_509: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_121, [1, 2048, 768]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_116: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_530, view_509);  convert_element_type_530 = view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_537: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_116, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_38: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_116);  add_116 = None
        any_39: "b8[]" = torch.ops.aten.any.default(isinf_38);  isinf_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_87: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43: "f32[]" = torch.ops.aten.where.self(any_39, full_default_87, full_default_86);  any_39 = full_default_87 = full_default_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_39: "f32[]" = torch.ops.aten.neg.default(where_43)
        clamp_min_38: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_537, neg_39);  convert_element_type_537 = neg_39 = None
        clamp_max_38: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_38, where_43);  clamp_min_38 = where_43 = None
        convert_element_type_538: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_38, torch.float16);  clamp_max_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_539: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_538, torch.float32)
        pow_41: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_539, 2);  convert_element_type_539 = None
        mean_40: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_117: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_83: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_538, rsqrt_40);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_540: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_83, torch.float16);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_84: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg166_1, convert_element_type_540);  arg166_1 = convert_element_type_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_510: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_84, [2048, 768])
        permute_234: "f16[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_122: "f16[2048, 768]" = torch.ops.aten.mm.default(view_510, permute_234);  view_510 = permute_234 = None
        view_511: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_122, [1, 2048, 768]);  mm_122 = None
        view_512: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_511, [1, 2048, -1, 64]);  view_511 = None
        permute_235: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_91: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_235, [1, 12, 2048, 64]);  permute_235 = None
        view_519: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_91, [12, 2048, 64]);  expand_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_513: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_84, [2048, 768])
        permute_236: "f16[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        mm_123: "f16[2048, 768]" = torch.ops.aten.mm.default(view_513, permute_236);  view_513 = permute_236 = None
        view_514: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_123, [1, 2048, 768]);  mm_123 = None
        view_515: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_514, [1, 2048, -1, 64]);  view_514 = None
        permute_237: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_240: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_237, [0, 1, 3, 2]);  permute_237 = None
        expand_92: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_240, [1, 12, 64, 2048]);  permute_240 = None
        view_520: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_92, [12, 64, 2048]);  expand_92 = None
        bmm_44: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_519, view_520);  view_519 = view_520 = None
        view_521: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_44, [1, 12, 2048, 2048]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_118: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_521, add_76);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_549: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32);  add_118 = None
        amax_22: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_549, [-1], True)
        sub_24: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_549, amax_22);  convert_element_type_549 = amax_22 = None
        exp_22: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_23: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_550: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_26, torch.float16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_93: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_550, [1, 12, 2048, 2048]);  convert_element_type_550 = None
        view_524: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_93, [12, 2048, 2048]);  expand_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_516: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_84, [2048, 768]);  mul_84 = None
        permute_238: "f16[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_124: "f16[2048, 768]" = torch.ops.aten.mm.default(view_516, permute_238);  view_516 = permute_238 = None
        view_517: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_124, [1, 2048, 768]);  mm_124 = None
        view_518: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_517, [1, 2048, -1, 64]);  view_517 = None
        permute_239: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_94: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_239, [1, 12, 2048, 64]);  permute_239 = None
        view_525: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_94, [12, 2048, 64]);  expand_94 = None
        bmm_45: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_524, view_525);  view_524 = view_525 = None
        view_526: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 12, 2048, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_241: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        clone_104: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_527: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_104, [1, 2048, -1]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_528: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_527, [2048, 768]);  view_527 = None
        permute_242: "f16[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_125: "f16[2048, 768]" = torch.ops.aten.mm.default(view_528, permute_242);  view_528 = permute_242 = None
        view_529: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_125, [1, 2048, 768]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_119: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_538, view_529);  convert_element_type_538 = view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_555: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_39: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_119);  add_119 = None
        any_40: "b8[]" = torch.ops.aten.any.default(isinf_39);  isinf_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_89: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: "f32[]" = torch.ops.aten.where.self(any_40, full_default_89, full_default_88);  any_40 = full_default_89 = full_default_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_40: "f32[]" = torch.ops.aten.neg.default(where_44)
        clamp_min_39: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_555, neg_40);  convert_element_type_555 = neg_40 = None
        clamp_max_39: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_39, where_44);  clamp_min_39 = where_44 = None
        convert_element_type_556: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_39, torch.float16);  clamp_max_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_557: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_556, torch.float32)
        pow_42: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_557, 2);  convert_element_type_557 = None
        mean_41: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_120: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_85: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_556, rsqrt_41);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_558: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_85, torch.float16);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_86: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg171_1, convert_element_type_558);  arg171_1 = convert_element_type_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_530: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_86, [2048, 768]);  mul_86 = None
        permute_243: "f16[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        mm_126: "f16[2048, 768]" = torch.ops.aten.mm.default(view_530, permute_243);  view_530 = permute_243 = None
        view_531: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_126, [1, 2048, 768]);  mm_126 = None
        view_532: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_531, [1, 2048, -1, 64]);  view_531 = None
        permute_244: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_95: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_244, [1, 12, 2048, 64]);  permute_244 = None
        view_539: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_95, [12, 2048, 64]);  expand_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_533: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_245: "f16[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_127: "f16[2048, 768]" = torch.ops.aten.mm.default(view_533, permute_245);  view_533 = permute_245 = None
        view_534: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_127, [1, 2048, 768]);  mm_127 = None
        view_535: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_534, [1, 2048, -1, 64]);  view_534 = None
        permute_246: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_249: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
        expand_96: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_249, [1, 12, 64, 2048]);  permute_249 = None
        view_540: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_96, [12, 64, 2048]);  expand_96 = None
        bmm_46: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_539, view_540);  view_539 = view_540 = None
        view_541: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_46, [1, 12, 2048, 2048]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_121: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_541, add_80);  view_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_567: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32);  add_121 = None
        amax_23: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_567, [-1], True)
        sub_25: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_567, amax_23);  convert_element_type_567 = amax_23 = None
        exp_23: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_24: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_568: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_27, torch.float16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_97: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_568, [1, 12, 2048, 2048]);  convert_element_type_568 = None
        view_544: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_97, [12, 2048, 2048]);  expand_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_536: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_247: "f16[768, 768]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_128: "f16[2048, 768]" = torch.ops.aten.mm.default(view_536, permute_247);  view_536 = permute_247 = None
        view_537: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_128, [1, 2048, 768]);  mm_128 = None
        view_538: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_537, [1, 2048, -1, 64]);  view_537 = None
        permute_248: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_98: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_248, [1, 12, 2048, 64]);  permute_248 = None
        view_545: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_98, [12, 2048, 64]);  expand_98 = None
        bmm_47: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_544, view_545);  view_544 = view_545 = None
        view_546: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 12, 2048, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_250: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        clone_107: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_547: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_107, [1, 2048, -1]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_548: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_547, [2048, 768]);  view_547 = None
        permute_251: "f16[768, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_129: "f16[2048, 768]" = torch.ops.aten.mm.default(view_548, permute_251);  view_548 = permute_251 = None
        view_549: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_129, [1, 2048, 768]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_122: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_556, view_549);  convert_element_type_556 = view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_573: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_122, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_40: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_122);  add_122 = None
        any_41: "b8[]" = torch.ops.aten.any.default(isinf_40);  isinf_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_91: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_45: "f32[]" = torch.ops.aten.where.self(any_41, full_default_91, full_default_90);  any_41 = full_default_91 = full_default_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_41: "f32[]" = torch.ops.aten.neg.default(where_45)
        clamp_min_40: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_573, neg_41);  convert_element_type_573 = neg_41 = None
        clamp_max_40: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_40, where_45);  clamp_min_40 = where_45 = None
        convert_element_type_574: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_40, torch.float16);  clamp_max_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_575: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_574, torch.float32)
        pow_43: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_575, 2);  convert_element_type_575 = None
        mean_42: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_123: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None
        rsqrt_42: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_87: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_574, rsqrt_42);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_576: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_87, torch.float16);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_88: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg176_1, convert_element_type_576);  arg176_1 = convert_element_type_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_550: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_88, [2048, 768]);  mul_88 = None
        permute_252: "f16[768, 3072]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_130: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_550, permute_252);  view_550 = permute_252 = None
        view_551: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_130, [1, 2048, 3072]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_17: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_551);  view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_552: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_17, [2048, 3072]);  relu_17 = None
        permute_253: "f16[3072, 768]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_131: "f16[2048, 768]" = torch.ops.aten.mm.default(view_552, permute_253);  view_552 = permute_253 = None
        view_553: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_131, [1, 2048, 768]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_124: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_574, view_553);  convert_element_type_574 = view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_581: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_41: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_124);  add_124 = None
        any_42: "b8[]" = torch.ops.aten.any.default(isinf_41);  isinf_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_93: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46: "f32[]" = torch.ops.aten.where.self(any_42, full_default_93, full_default_92);  any_42 = full_default_93 = full_default_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_42: "f32[]" = torch.ops.aten.neg.default(where_46)
        clamp_min_41: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_581, neg_42);  convert_element_type_581 = neg_42 = None
        clamp_max_41: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_41, where_46);  clamp_min_41 = where_46 = None
        convert_element_type_582: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_41, torch.float16);  clamp_max_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_583: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_582, torch.float32)
        pow_44: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_583, 2);  convert_element_type_583 = None
        mean_43: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_125: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-06);  mean_43 = None
        rsqrt_43: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_89: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_582, rsqrt_43);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_584: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_89, torch.float16);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_90: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg179_1, convert_element_type_584);  arg179_1 = convert_element_type_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_554: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_90, [2048, 768])
        permute_254: "f16[768, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_132: "f16[2048, 768]" = torch.ops.aten.mm.default(view_554, permute_254);  view_554 = permute_254 = None
        view_555: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_132, [1, 2048, 768]);  mm_132 = None
        view_556: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_555, [1, 2048, -1, 64]);  view_555 = None
        permute_255: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_99: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_255, [1, 12, 2048, 64]);  permute_255 = None
        view_563: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_99, [12, 2048, 64]);  expand_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_557: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_90, [2048, 768])
        permute_256: "f16[768, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_133: "f16[2048, 768]" = torch.ops.aten.mm.default(view_557, permute_256);  view_557 = permute_256 = None
        view_558: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_133, [1, 2048, 768]);  mm_133 = None
        view_559: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_558, [1, 2048, -1, 64]);  view_558 = None
        permute_257: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_260: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_100: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_260, [1, 12, 64, 2048]);  permute_260 = None
        view_564: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_100, [12, 64, 2048]);  expand_100 = None
        bmm_48: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_563, view_564);  view_563 = view_564 = None
        view_565: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_48, [1, 12, 2048, 2048]);  bmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_126: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_565, add_76);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_593: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_126, torch.float32);  add_126 = None
        amax_24: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_593, [-1], True)
        sub_26: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_593, amax_24);  convert_element_type_593 = amax_24 = None
        exp_24: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_25: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_28: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        convert_element_type_594: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_28, torch.float16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_101: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_594, [1, 12, 2048, 2048]);  convert_element_type_594 = None
        view_568: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_101, [12, 2048, 2048]);  expand_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_560: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_90, [2048, 768]);  mul_90 = None
        permute_258: "f16[768, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_134: "f16[2048, 768]" = torch.ops.aten.mm.default(view_560, permute_258);  view_560 = permute_258 = None
        view_561: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_134, [1, 2048, 768]);  mm_134 = None
        view_562: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_561, [1, 2048, -1, 64]);  view_561 = None
        permute_259: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_102: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_259, [1, 12, 2048, 64]);  permute_259 = None
        view_569: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_102, [12, 2048, 64]);  expand_102 = None
        bmm_49: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_568, view_569);  view_568 = view_569 = None
        view_570: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_49, [1, 12, 2048, 64]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_261: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        clone_112: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_571: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_112, [1, 2048, -1]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_572: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_571, [2048, 768]);  view_571 = None
        permute_262: "f16[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_135: "f16[2048, 768]" = torch.ops.aten.mm.default(view_572, permute_262);  view_572 = permute_262 = None
        view_573: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_135, [1, 2048, 768]);  mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_127: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_582, view_573);  convert_element_type_582 = view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_599: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_42: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_127);  add_127 = None
        any_43: "b8[]" = torch.ops.aten.any.default(isinf_42);  isinf_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_95: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47: "f32[]" = torch.ops.aten.where.self(any_43, full_default_95, full_default_94);  any_43 = full_default_95 = full_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_43: "f32[]" = torch.ops.aten.neg.default(where_47)
        clamp_min_42: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_599, neg_43);  convert_element_type_599 = neg_43 = None
        clamp_max_42: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_42, where_47);  clamp_min_42 = where_47 = None
        convert_element_type_600: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_42, torch.float16);  clamp_max_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_601: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_600, torch.float32)
        pow_45: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 2);  convert_element_type_601 = None
        mean_44: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_128: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_44, 1e-06);  mean_44 = None
        rsqrt_44: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_91: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_600, rsqrt_44);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_602: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_91, torch.float16);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_92: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg184_1, convert_element_type_602);  arg184_1 = convert_element_type_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_574: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_92, [2048, 768]);  mul_92 = None
        permute_263: "f16[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_136: "f16[2048, 768]" = torch.ops.aten.mm.default(view_574, permute_263);  view_574 = permute_263 = None
        view_575: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_136, [1, 2048, 768]);  mm_136 = None
        view_576: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_575, [1, 2048, -1, 64]);  view_575 = None
        permute_264: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_103: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_264, [1, 12, 2048, 64]);  permute_264 = None
        view_583: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_103, [12, 2048, 64]);  expand_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_577: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_265: "f16[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        mm_137: "f16[2048, 768]" = torch.ops.aten.mm.default(view_577, permute_265);  view_577 = permute_265 = None
        view_578: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_137, [1, 2048, 768]);  mm_137 = None
        view_579: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_578, [1, 2048, -1, 64]);  view_578 = None
        permute_266: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_269: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_266, [0, 1, 3, 2]);  permute_266 = None
        expand_104: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_269, [1, 12, 64, 2048]);  permute_269 = None
        view_584: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_104, [12, 64, 2048]);  expand_104 = None
        bmm_50: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_583, view_584);  view_583 = view_584 = None
        view_585: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_50, [1, 12, 2048, 2048]);  bmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_129: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_585, add_80);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_611: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        amax_25: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_611, [-1], True)
        sub_27: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_611, amax_25);  convert_element_type_611 = amax_25 = None
        exp_25: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_26: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_29: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        convert_element_type_612: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_29, torch.float16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_105: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_612, [1, 12, 2048, 2048]);  convert_element_type_612 = None
        view_588: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_105, [12, 2048, 2048]);  expand_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_580: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_267: "f16[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_138: "f16[2048, 768]" = torch.ops.aten.mm.default(view_580, permute_267);  view_580 = permute_267 = None
        view_581: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_138, [1, 2048, 768]);  mm_138 = None
        view_582: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_581, [1, 2048, -1, 64]);  view_581 = None
        permute_268: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_106: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_268, [1, 12, 2048, 64]);  permute_268 = None
        view_589: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_106, [12, 2048, 64]);  expand_106 = None
        bmm_51: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_588, view_589);  view_588 = view_589 = None
        view_590: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_51, [1, 12, 2048, 64]);  bmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_270: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None
        clone_115: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_591: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_115, [1, 2048, -1]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_592: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_591, [2048, 768]);  view_591 = None
        permute_271: "f16[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_139: "f16[2048, 768]" = torch.ops.aten.mm.default(view_592, permute_271);  view_592 = permute_271 = None
        view_593: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_139, [1, 2048, 768]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_130: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_600, view_593);  convert_element_type_600 = view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_617: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_130, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_43: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_130);  add_130 = None
        any_44: "b8[]" = torch.ops.aten.any.default(isinf_43);  isinf_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_97: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: "f32[]" = torch.ops.aten.where.self(any_44, full_default_97, full_default_96);  any_44 = full_default_97 = full_default_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_44: "f32[]" = torch.ops.aten.neg.default(where_48)
        clamp_min_43: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_617, neg_44);  convert_element_type_617 = neg_44 = None
        clamp_max_43: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_43, where_48);  clamp_min_43 = where_48 = None
        convert_element_type_618: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_43, torch.float16);  clamp_max_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_619: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_618, torch.float32)
        pow_46: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_619, 2);  convert_element_type_619 = None
        mean_45: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_131: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-06);  mean_45 = None
        rsqrt_45: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_93: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_618, rsqrt_45);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_620: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_93, torch.float16);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_94: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg189_1, convert_element_type_620);  arg189_1 = convert_element_type_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_594: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_94, [2048, 768]);  mul_94 = None
        permute_272: "f16[768, 3072]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        mm_140: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_594, permute_272);  view_594 = permute_272 = None
        view_595: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_140, [1, 2048, 3072]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_18: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_595);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_596: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_18, [2048, 3072]);  relu_18 = None
        permute_273: "f16[3072, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_141: "f16[2048, 768]" = torch.ops.aten.mm.default(view_596, permute_273);  view_596 = permute_273 = None
        view_597: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_141, [1, 2048, 768]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_132: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_618, view_597);  convert_element_type_618 = view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_625: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_44: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_132);  add_132 = None
        any_45: "b8[]" = torch.ops.aten.any.default(isinf_44);  isinf_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_99: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49: "f32[]" = torch.ops.aten.where.self(any_45, full_default_99, full_default_98);  any_45 = full_default_99 = full_default_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_45: "f32[]" = torch.ops.aten.neg.default(where_49)
        clamp_min_44: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_625, neg_45);  convert_element_type_625 = neg_45 = None
        clamp_max_44: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_44, where_49);  clamp_min_44 = where_49 = None
        convert_element_type_626: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_44, torch.float16);  clamp_max_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_627: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_626, torch.float32)
        pow_47: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_627, 2);  convert_element_type_627 = None
        mean_46: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_46, 1e-06);  mean_46 = None
        rsqrt_46: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_95: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_626, rsqrt_46);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_628: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_95, torch.float16);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_96: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg192_1, convert_element_type_628);  arg192_1 = convert_element_type_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_598: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_96, [2048, 768])
        permute_274: "f16[768, 768]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        mm_142: "f16[2048, 768]" = torch.ops.aten.mm.default(view_598, permute_274);  view_598 = permute_274 = None
        view_599: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_142, [1, 2048, 768]);  mm_142 = None
        view_600: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_599, [1, 2048, -1, 64]);  view_599 = None
        permute_275: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_107: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_275, [1, 12, 2048, 64]);  permute_275 = None
        view_607: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_107, [12, 2048, 64]);  expand_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_601: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_96, [2048, 768])
        permute_276: "f16[768, 768]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        mm_143: "f16[2048, 768]" = torch.ops.aten.mm.default(view_601, permute_276);  view_601 = permute_276 = None
        view_602: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_143, [1, 2048, 768]);  mm_143 = None
        view_603: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_602, [1, 2048, -1, 64]);  view_602 = None
        permute_277: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_280: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_277, [0, 1, 3, 2]);  permute_277 = None
        expand_108: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_280, [1, 12, 64, 2048]);  permute_280 = None
        view_608: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_108, [12, 64, 2048]);  expand_108 = None
        bmm_52: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_607, view_608);  view_607 = view_608 = None
        view_609: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_52, [1, 12, 2048, 2048]);  bmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_134: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_609, add_76);  view_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_637: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32);  add_134 = None
        amax_26: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_637, [-1], True)
        sub_28: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_637, amax_26);  convert_element_type_637 = amax_26 = None
        exp_26: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_27: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_30: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        convert_element_type_638: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_30, torch.float16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_109: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_638, [1, 12, 2048, 2048]);  convert_element_type_638 = None
        view_612: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_109, [12, 2048, 2048]);  expand_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_604: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_96, [2048, 768]);  mul_96 = None
        permute_278: "f16[768, 768]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_144: "f16[2048, 768]" = torch.ops.aten.mm.default(view_604, permute_278);  view_604 = permute_278 = None
        view_605: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_144, [1, 2048, 768]);  mm_144 = None
        view_606: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_605, [1, 2048, -1, 64]);  view_605 = None
        permute_279: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_110: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_279, [1, 12, 2048, 64]);  permute_279 = None
        view_613: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_110, [12, 2048, 64]);  expand_110 = None
        bmm_53: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_612, view_613);  view_612 = view_613 = None
        view_614: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_53, [1, 12, 2048, 64]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_281: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        clone_120: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_615: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_120, [1, 2048, -1]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_616: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_615, [2048, 768]);  view_615 = None
        permute_282: "f16[768, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        mm_145: "f16[2048, 768]" = torch.ops.aten.mm.default(view_616, permute_282);  view_616 = permute_282 = None
        view_617: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_145, [1, 2048, 768]);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_135: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_626, view_617);  convert_element_type_626 = view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_643: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_45: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_135);  add_135 = None
        any_46: "b8[]" = torch.ops.aten.any.default(isinf_45);  isinf_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_101: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_50: "f32[]" = torch.ops.aten.where.self(any_46, full_default_101, full_default_100);  any_46 = full_default_101 = full_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_46: "f32[]" = torch.ops.aten.neg.default(where_50)
        clamp_min_45: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_643, neg_46);  convert_element_type_643 = neg_46 = None
        clamp_max_45: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_45, where_50);  clamp_min_45 = where_50 = None
        convert_element_type_644: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_45, torch.float16);  clamp_max_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_645: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_644, torch.float32)
        pow_48: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_645, 2);  convert_element_type_645 = None
        mean_47: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-06);  mean_47 = None
        rsqrt_47: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_97: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_644, rsqrt_47);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_646: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_97, torch.float16);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_98: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg197_1, convert_element_type_646);  arg197_1 = convert_element_type_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_618: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_98, [2048, 768]);  mul_98 = None
        permute_283: "f16[768, 768]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        mm_146: "f16[2048, 768]" = torch.ops.aten.mm.default(view_618, permute_283);  view_618 = permute_283 = None
        view_619: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_146, [1, 2048, 768]);  mm_146 = None
        view_620: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_619, [1, 2048, -1, 64]);  view_619 = None
        permute_284: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_111: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_284, [1, 12, 2048, 64]);  permute_284 = None
        view_627: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_111, [12, 2048, 64]);  expand_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_621: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_285: "f16[768, 768]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_147: "f16[2048, 768]" = torch.ops.aten.mm.default(view_621, permute_285);  view_621 = permute_285 = None
        view_622: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_147, [1, 2048, 768]);  mm_147 = None
        view_623: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_622, [1, 2048, -1, 64]);  view_622 = None
        permute_286: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_289: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_286, [0, 1, 3, 2]);  permute_286 = None
        expand_112: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_289, [1, 12, 64, 2048]);  permute_289 = None
        view_628: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_112, [12, 64, 2048]);  expand_112 = None
        bmm_54: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_627, view_628);  view_627 = view_628 = None
        view_629: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_54, [1, 12, 2048, 2048]);  bmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_137: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_629, add_80);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_655: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_137, torch.float32);  add_137 = None
        amax_27: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_655, [-1], True)
        sub_29: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_655, amax_27);  convert_element_type_655 = amax_27 = None
        exp_27: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_28: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_31: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        convert_element_type_656: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_31, torch.float16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_113: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_656, [1, 12, 2048, 2048]);  convert_element_type_656 = None
        view_632: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_113, [12, 2048, 2048]);  expand_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_624: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_287: "f16[768, 768]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        mm_148: "f16[2048, 768]" = torch.ops.aten.mm.default(view_624, permute_287);  view_624 = permute_287 = None
        view_625: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_148, [1, 2048, 768]);  mm_148 = None
        view_626: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_625, [1, 2048, -1, 64]);  view_625 = None
        permute_288: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_114: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_288, [1, 12, 2048, 64]);  permute_288 = None
        view_633: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_114, [12, 2048, 64]);  expand_114 = None
        bmm_55: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_632, view_633);  view_632 = view_633 = None
        view_634: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_55, [1, 12, 2048, 64]);  bmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_290: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
        clone_123: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_635: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_123, [1, 2048, -1]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_636: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_635, [2048, 768]);  view_635 = None
        permute_291: "f16[768, 768]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        mm_149: "f16[2048, 768]" = torch.ops.aten.mm.default(view_636, permute_291);  view_636 = permute_291 = None
        view_637: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_149, [1, 2048, 768]);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_138: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_644, view_637);  convert_element_type_644 = view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_661: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_138, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_46: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_138);  add_138 = None
        any_47: "b8[]" = torch.ops.aten.any.default(isinf_46);  isinf_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_103: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_51: "f32[]" = torch.ops.aten.where.self(any_47, full_default_103, full_default_102);  any_47 = full_default_103 = full_default_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_47: "f32[]" = torch.ops.aten.neg.default(where_51)
        clamp_min_46: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_661, neg_47);  convert_element_type_661 = neg_47 = None
        clamp_max_46: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_46, where_51);  clamp_min_46 = where_51 = None
        convert_element_type_662: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_46, torch.float16);  clamp_max_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_663: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_662, torch.float32)
        pow_49: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_663, 2);  convert_element_type_663 = None
        mean_48: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_48, 1e-06);  mean_48 = None
        rsqrt_48: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_99: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_662, rsqrt_48);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_664: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_99, torch.float16);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_100: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg202_1, convert_element_type_664);  arg202_1 = convert_element_type_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_638: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_100, [2048, 768]);  mul_100 = None
        permute_292: "f16[768, 3072]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        mm_150: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_638, permute_292);  view_638 = permute_292 = None
        view_639: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_150, [1, 2048, 3072]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_19: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_639);  view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_640: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_19, [2048, 3072]);  relu_19 = None
        permute_293: "f16[3072, 768]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_151: "f16[2048, 768]" = torch.ops.aten.mm.default(view_640, permute_293);  view_640 = permute_293 = None
        view_641: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_151, [1, 2048, 768]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_140: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_662, view_641);  convert_element_type_662 = view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_669: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_140, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_47: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_140);  add_140 = None
        any_48: "b8[]" = torch.ops.aten.any.default(isinf_47);  isinf_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_105: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_52: "f32[]" = torch.ops.aten.where.self(any_48, full_default_105, full_default_104);  any_48 = full_default_105 = full_default_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_48: "f32[]" = torch.ops.aten.neg.default(where_52)
        clamp_min_47: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_669, neg_48);  convert_element_type_669 = neg_48 = None
        clamp_max_47: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_47, where_52);  clamp_min_47 = where_52 = None
        convert_element_type_670: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_47, torch.float16);  clamp_max_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_671: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_670, torch.float32)
        pow_50: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_671, 2);  convert_element_type_671 = None
        mean_49: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_141: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-06);  mean_49 = None
        rsqrt_49: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_101: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_670, rsqrt_49);  rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_672: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_101, torch.float16);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_102: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg205_1, convert_element_type_672);  arg205_1 = convert_element_type_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_642: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_102, [2048, 768])
        permute_294: "f16[768, 768]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        mm_152: "f16[2048, 768]" = torch.ops.aten.mm.default(view_642, permute_294);  view_642 = permute_294 = None
        view_643: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_152, [1, 2048, 768]);  mm_152 = None
        view_644: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_643, [1, 2048, -1, 64]);  view_643 = None
        permute_295: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_115: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_295, [1, 12, 2048, 64]);  permute_295 = None
        view_651: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_115, [12, 2048, 64]);  expand_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_645: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_102, [2048, 768])
        permute_296: "f16[768, 768]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        mm_153: "f16[2048, 768]" = torch.ops.aten.mm.default(view_645, permute_296);  view_645 = permute_296 = None
        view_646: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_153, [1, 2048, 768]);  mm_153 = None
        view_647: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_646, [1, 2048, -1, 64]);  view_646 = None
        permute_297: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_647, [0, 2, 1, 3]);  view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_300: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_297, [0, 1, 3, 2]);  permute_297 = None
        expand_116: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_300, [1, 12, 64, 2048]);  permute_300 = None
        view_652: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_116, [12, 64, 2048]);  expand_116 = None
        bmm_56: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_651, view_652);  view_651 = view_652 = None
        view_653: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_56, [1, 12, 2048, 2048]);  bmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_142: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_653, add_76);  view_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_681: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_142, torch.float32);  add_142 = None
        amax_28: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_681, [-1], True)
        sub_30: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_681, amax_28);  convert_element_type_681 = amax_28 = None
        exp_28: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_29: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_32: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        convert_element_type_682: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_32, torch.float16);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_117: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_682, [1, 12, 2048, 2048]);  convert_element_type_682 = None
        view_656: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_117, [12, 2048, 2048]);  expand_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_648: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_102, [2048, 768]);  mul_102 = None
        permute_298: "f16[768, 768]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        mm_154: "f16[2048, 768]" = torch.ops.aten.mm.default(view_648, permute_298);  view_648 = permute_298 = None
        view_649: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_154, [1, 2048, 768]);  mm_154 = None
        view_650: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_649, [1, 2048, -1, 64]);  view_649 = None
        permute_299: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_118: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_299, [1, 12, 2048, 64]);  permute_299 = None
        view_657: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_118, [12, 2048, 64]);  expand_118 = None
        bmm_57: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_656, view_657);  view_656 = view_657 = None
        view_658: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_57, [1, 12, 2048, 64]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_301: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
        clone_128: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_301, memory_format = torch.contiguous_format);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_659: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_128, [1, 2048, -1]);  clone_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_660: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_659, [2048, 768]);  view_659 = None
        permute_302: "f16[768, 768]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_155: "f16[2048, 768]" = torch.ops.aten.mm.default(view_660, permute_302);  view_660 = permute_302 = None
        view_661: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_155, [1, 2048, 768]);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_143: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_670, view_661);  convert_element_type_670 = view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_687: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_143, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_48: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_143);  add_143 = None
        any_49: "b8[]" = torch.ops.aten.any.default(isinf_48);  isinf_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_107: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_53: "f32[]" = torch.ops.aten.where.self(any_49, full_default_107, full_default_106);  any_49 = full_default_107 = full_default_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_49: "f32[]" = torch.ops.aten.neg.default(where_53)
        clamp_min_48: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_687, neg_49);  convert_element_type_687 = neg_49 = None
        clamp_max_48: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_48, where_53);  clamp_min_48 = where_53 = None
        convert_element_type_688: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_48, torch.float16);  clamp_max_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_689: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_688, torch.float32)
        pow_51: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_689, 2);  convert_element_type_689 = None
        mean_50: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_144: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_50, 1e-06);  mean_50 = None
        rsqrt_50: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_103: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_688, rsqrt_50);  rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_690: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_103, torch.float16);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_104: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg210_1, convert_element_type_690);  arg210_1 = convert_element_type_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_662: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_104, [2048, 768]);  mul_104 = None
        permute_303: "f16[768, 768]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        mm_156: "f16[2048, 768]" = torch.ops.aten.mm.default(view_662, permute_303);  view_662 = permute_303 = None
        view_663: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_156, [1, 2048, 768]);  mm_156 = None
        view_664: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_663, [1, 2048, -1, 64]);  view_663 = None
        permute_304: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_119: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_304, [1, 12, 2048, 64]);  permute_304 = None
        view_671: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_119, [12, 2048, 64]);  expand_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_665: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_305: "f16[768, 768]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        mm_157: "f16[2048, 768]" = torch.ops.aten.mm.default(view_665, permute_305);  view_665 = permute_305 = None
        view_666: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_157, [1, 2048, 768]);  mm_157 = None
        view_667: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_666, [1, 2048, -1, 64]);  view_666 = None
        permute_306: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_667, [0, 2, 1, 3]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_309: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_306, [0, 1, 3, 2]);  permute_306 = None
        expand_120: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_309, [1, 12, 64, 2048]);  permute_309 = None
        view_672: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_120, [12, 64, 2048]);  expand_120 = None
        bmm_58: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_671, view_672);  view_671 = view_672 = None
        view_673: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_58, [1, 12, 2048, 2048]);  bmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_145: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_673, add_80);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_699: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_145, torch.float32);  add_145 = None
        amax_29: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_699, [-1], True)
        sub_31: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_699, amax_29);  convert_element_type_699 = amax_29 = None
        exp_29: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_30: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_33: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        convert_element_type_700: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_33, torch.float16);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_121: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_700, [1, 12, 2048, 2048]);  convert_element_type_700 = None
        view_676: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_121, [12, 2048, 2048]);  expand_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_668: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_307: "f16[768, 768]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        mm_158: "f16[2048, 768]" = torch.ops.aten.mm.default(view_668, permute_307);  view_668 = permute_307 = None
        view_669: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_158, [1, 2048, 768]);  mm_158 = None
        view_670: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_669, [1, 2048, -1, 64]);  view_669 = None
        permute_308: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_122: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_308, [1, 12, 2048, 64]);  permute_308 = None
        view_677: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_122, [12, 2048, 64]);  expand_122 = None
        bmm_59: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_676, view_677);  view_676 = view_677 = None
        view_678: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_59, [1, 12, 2048, 64]);  bmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_310: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_678, [0, 2, 1, 3]);  view_678 = None
        clone_131: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_679: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_131, [1, 2048, -1]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_680: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_679, [2048, 768]);  view_679 = None
        permute_311: "f16[768, 768]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        mm_159: "f16[2048, 768]" = torch.ops.aten.mm.default(view_680, permute_311);  view_680 = permute_311 = None
        view_681: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_159, [1, 2048, 768]);  mm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_146: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_688, view_681);  convert_element_type_688 = view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_705: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_49: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_146);  add_146 = None
        any_50: "b8[]" = torch.ops.aten.any.default(isinf_49);  isinf_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_109: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_54: "f32[]" = torch.ops.aten.where.self(any_50, full_default_109, full_default_108);  any_50 = full_default_109 = full_default_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_50: "f32[]" = torch.ops.aten.neg.default(where_54)
        clamp_min_49: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_705, neg_50);  convert_element_type_705 = neg_50 = None
        clamp_max_49: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_49, where_54);  clamp_min_49 = where_54 = None
        convert_element_type_706: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_49, torch.float16);  clamp_max_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_707: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_706, torch.float32)
        pow_52: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_707, 2);  convert_element_type_707 = None
        mean_51: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_147: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_51, 1e-06);  mean_51 = None
        rsqrt_51: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_105: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_706, rsqrt_51);  rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_708: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_105, torch.float16);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_106: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg215_1, convert_element_type_708);  arg215_1 = convert_element_type_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_682: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_106, [2048, 768]);  mul_106 = None
        permute_312: "f16[768, 3072]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        mm_160: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_682, permute_312);  view_682 = permute_312 = None
        view_683: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_160, [1, 2048, 3072]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_20: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_683);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_684: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_20, [2048, 3072]);  relu_20 = None
        permute_313: "f16[3072, 768]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_161: "f16[2048, 768]" = torch.ops.aten.mm.default(view_684, permute_313);  view_684 = permute_313 = None
        view_685: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_161, [1, 2048, 768]);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_148: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_706, view_685);  convert_element_type_706 = view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_713: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_148, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_50: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_148);  add_148 = None
        any_51: "b8[]" = torch.ops.aten.any.default(isinf_50);  isinf_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_111: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_110: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_55: "f32[]" = torch.ops.aten.where.self(any_51, full_default_111, full_default_110);  any_51 = full_default_111 = full_default_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_51: "f32[]" = torch.ops.aten.neg.default(where_55)
        clamp_min_50: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_713, neg_51);  convert_element_type_713 = neg_51 = None
        clamp_max_50: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_50, where_55);  clamp_min_50 = where_55 = None
        convert_element_type_714: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_50, torch.float16);  clamp_max_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_715: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_714, torch.float32)
        pow_53: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 2);  convert_element_type_715 = None
        mean_52: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_149: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_52, 1e-06);  mean_52 = None
        rsqrt_52: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_107: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_714, rsqrt_52);  rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_716: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_107, torch.float16);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_108: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg218_1, convert_element_type_716);  arg218_1 = convert_element_type_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_686: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_108, [2048, 768])
        permute_314: "f16[768, 768]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        mm_162: "f16[2048, 768]" = torch.ops.aten.mm.default(view_686, permute_314);  view_686 = permute_314 = None
        view_687: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_162, [1, 2048, 768]);  mm_162 = None
        view_688: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_687, [1, 2048, -1, 64]);  view_687 = None
        permute_315: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_123: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_315, [1, 12, 2048, 64]);  permute_315 = None
        view_695: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_123, [12, 2048, 64]);  expand_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_689: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_108, [2048, 768])
        permute_316: "f16[768, 768]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_163: "f16[2048, 768]" = torch.ops.aten.mm.default(view_689, permute_316);  view_689 = permute_316 = None
        view_690: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_163, [1, 2048, 768]);  mm_163 = None
        view_691: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_690, [1, 2048, -1, 64]);  view_690 = None
        permute_317: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_320: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_317, [0, 1, 3, 2]);  permute_317 = None
        expand_124: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_320, [1, 12, 64, 2048]);  permute_320 = None
        view_696: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_124, [12, 64, 2048]);  expand_124 = None
        bmm_60: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_695, view_696);  view_695 = view_696 = None
        view_697: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_60, [1, 12, 2048, 2048]);  bmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_150: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_697, add_76);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_725: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_150, torch.float32);  add_150 = None
        amax_30: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_725, [-1], True)
        sub_32: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_725, amax_30);  convert_element_type_725 = amax_30 = None
        exp_30: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_31: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_34: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        convert_element_type_726: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_34, torch.float16);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_125: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_726, [1, 12, 2048, 2048]);  convert_element_type_726 = None
        view_700: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_125, [12, 2048, 2048]);  expand_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_692: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_108, [2048, 768]);  mul_108 = None
        permute_318: "f16[768, 768]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        mm_164: "f16[2048, 768]" = torch.ops.aten.mm.default(view_692, permute_318);  view_692 = permute_318 = None
        view_693: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_164, [1, 2048, 768]);  mm_164 = None
        view_694: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_693, [1, 2048, -1, 64]);  view_693 = None
        permute_319: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_126: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_319, [1, 12, 2048, 64]);  permute_319 = None
        view_701: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_126, [12, 2048, 64]);  expand_126 = None
        bmm_61: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_700, view_701);  view_700 = view_701 = None
        view_702: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_61, [1, 12, 2048, 64]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_321: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_702, [0, 2, 1, 3]);  view_702 = None
        clone_136: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_703: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_136, [1, 2048, -1]);  clone_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_704: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_703, [2048, 768]);  view_703 = None
        permute_322: "f16[768, 768]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        mm_165: "f16[2048, 768]" = torch.ops.aten.mm.default(view_704, permute_322);  view_704 = permute_322 = None
        view_705: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_165, [1, 2048, 768]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_151: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_714, view_705);  convert_element_type_714 = view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_731: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_151, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_51: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_151);  add_151 = None
        any_52: "b8[]" = torch.ops.aten.any.default(isinf_51);  isinf_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_113: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_112: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_56: "f32[]" = torch.ops.aten.where.self(any_52, full_default_113, full_default_112);  any_52 = full_default_113 = full_default_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_52: "f32[]" = torch.ops.aten.neg.default(where_56)
        clamp_min_51: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_731, neg_52);  convert_element_type_731 = neg_52 = None
        clamp_max_51: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_51, where_56);  clamp_min_51 = where_56 = None
        convert_element_type_732: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_51, torch.float16);  clamp_max_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_733: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_732, torch.float32)
        pow_54: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_733, 2);  convert_element_type_733 = None
        mean_53: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_152: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_53, 1e-06);  mean_53 = None
        rsqrt_53: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_109: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_732, rsqrt_53);  rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_734: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_109, torch.float16);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_110: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg223_1, convert_element_type_734);  arg223_1 = convert_element_type_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_706: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_110, [2048, 768]);  mul_110 = None
        permute_323: "f16[768, 768]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        mm_166: "f16[2048, 768]" = torch.ops.aten.mm.default(view_706, permute_323);  view_706 = permute_323 = None
        view_707: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_166, [1, 2048, 768]);  mm_166 = None
        view_708: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_707, [1, 2048, -1, 64]);  view_707 = None
        permute_324: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_127: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_324, [1, 12, 2048, 64]);  permute_324 = None
        view_715: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_127, [12, 2048, 64]);  expand_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_709: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_325: "f16[768, 768]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        mm_167: "f16[2048, 768]" = torch.ops.aten.mm.default(view_709, permute_325);  view_709 = permute_325 = None
        view_710: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_167, [1, 2048, 768]);  mm_167 = None
        view_711: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_710, [1, 2048, -1, 64]);  view_710 = None
        permute_326: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_711, [0, 2, 1, 3]);  view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_329: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_326, [0, 1, 3, 2]);  permute_326 = None
        expand_128: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_329, [1, 12, 64, 2048]);  permute_329 = None
        view_716: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_128, [12, 64, 2048]);  expand_128 = None
        bmm_62: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_715, view_716);  view_715 = view_716 = None
        view_717: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_62, [1, 12, 2048, 2048]);  bmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_153: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_717, add_80);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_743: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_153, torch.float32);  add_153 = None
        amax_31: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_743, [-1], True)
        sub_33: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_743, amax_31);  convert_element_type_743 = amax_31 = None
        exp_31: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_32: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_35: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        convert_element_type_744: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_35, torch.float16);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_129: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_744, [1, 12, 2048, 2048]);  convert_element_type_744 = None
        view_720: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_129, [12, 2048, 2048]);  expand_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_712: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_327: "f16[768, 768]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_168: "f16[2048, 768]" = torch.ops.aten.mm.default(view_712, permute_327);  view_712 = permute_327 = None
        view_713: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_168, [1, 2048, 768]);  mm_168 = None
        view_714: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_713, [1, 2048, -1, 64]);  view_713 = None
        permute_328: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_130: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_328, [1, 12, 2048, 64]);  permute_328 = None
        view_721: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_130, [12, 2048, 64]);  expand_130 = None
        bmm_63: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_720, view_721);  view_720 = view_721 = None
        view_722: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_63, [1, 12, 2048, 64]);  bmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_330: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None
        clone_139: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_723: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_139, [1, 2048, -1]);  clone_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_724: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_723, [2048, 768]);  view_723 = None
        permute_331: "f16[768, 768]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_169: "f16[2048, 768]" = torch.ops.aten.mm.default(view_724, permute_331);  view_724 = permute_331 = None
        view_725: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_169, [1, 2048, 768]);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_154: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_732, view_725);  convert_element_type_732 = view_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_749: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_154, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_52: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_154);  add_154 = None
        any_53: "b8[]" = torch.ops.aten.any.default(isinf_52);  isinf_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_115: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_114: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_57: "f32[]" = torch.ops.aten.where.self(any_53, full_default_115, full_default_114);  any_53 = full_default_115 = full_default_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_53: "f32[]" = torch.ops.aten.neg.default(where_57)
        clamp_min_52: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_749, neg_53);  convert_element_type_749 = neg_53 = None
        clamp_max_52: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_52, where_57);  clamp_min_52 = where_57 = None
        convert_element_type_750: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_52, torch.float16);  clamp_max_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_751: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_750, torch.float32)
        pow_55: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_751, 2);  convert_element_type_751 = None
        mean_54: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_155: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_54, 1e-06);  mean_54 = None
        rsqrt_54: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_111: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_750, rsqrt_54);  rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_752: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_111, torch.float16);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_112: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg228_1, convert_element_type_752);  arg228_1 = convert_element_type_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_726: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_112, [2048, 768]);  mul_112 = None
        permute_332: "f16[768, 3072]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_170: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_726, permute_332);  view_726 = permute_332 = None
        view_727: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_170, [1, 2048, 3072]);  mm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_21: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_727);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_728: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_21, [2048, 3072]);  relu_21 = None
        permute_333: "f16[3072, 768]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        mm_171: "f16[2048, 768]" = torch.ops.aten.mm.default(view_728, permute_333);  view_728 = permute_333 = None
        view_729: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_171, [1, 2048, 768]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_156: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_750, view_729);  convert_element_type_750 = view_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_757: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_156, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_53: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_156);  add_156 = None
        any_54: "b8[]" = torch.ops.aten.any.default(isinf_53);  isinf_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_117: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_58: "f32[]" = torch.ops.aten.where.self(any_54, full_default_117, full_default_116);  any_54 = full_default_117 = full_default_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_54: "f32[]" = torch.ops.aten.neg.default(where_58)
        clamp_min_53: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_757, neg_54);  convert_element_type_757 = neg_54 = None
        clamp_max_53: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_53, where_58);  clamp_min_53 = where_58 = None
        convert_element_type_758: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_53, torch.float16);  clamp_max_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_759: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_758, torch.float32)
        pow_56: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_759, 2);  convert_element_type_759 = None
        mean_55: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_157: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_55, 1e-06);  mean_55 = None
        rsqrt_55: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_113: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_758, rsqrt_55);  rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_760: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_113, torch.float16);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_114: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg231_1, convert_element_type_760);  arg231_1 = convert_element_type_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_730: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_114, [2048, 768])
        permute_334: "f16[768, 768]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        mm_172: "f16[2048, 768]" = torch.ops.aten.mm.default(view_730, permute_334);  view_730 = permute_334 = None
        view_731: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_172, [1, 2048, 768]);  mm_172 = None
        view_732: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_731, [1, 2048, -1, 64]);  view_731 = None
        permute_335: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_131: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_335, [1, 12, 2048, 64]);  permute_335 = None
        view_739: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_131, [12, 2048, 64]);  expand_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_733: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_114, [2048, 768])
        permute_336: "f16[768, 768]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        mm_173: "f16[2048, 768]" = torch.ops.aten.mm.default(view_733, permute_336);  view_733 = permute_336 = None
        view_734: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_173, [1, 2048, 768]);  mm_173 = None
        view_735: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_734, [1, 2048, -1, 64]);  view_734 = None
        permute_337: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_340: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_337, [0, 1, 3, 2]);  permute_337 = None
        expand_132: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_340, [1, 12, 64, 2048]);  permute_340 = None
        view_740: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_132, [12, 64, 2048]);  expand_132 = None
        bmm_64: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_739, view_740);  view_739 = view_740 = None
        view_741: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_64, [1, 12, 2048, 2048]);  bmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_158: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_741, add_76);  view_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_769: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_158, torch.float32);  add_158 = None
        amax_32: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_769, [-1], True)
        sub_34: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_769, amax_32);  convert_element_type_769 = amax_32 = None
        exp_32: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_33: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_36: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        convert_element_type_770: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_36, torch.float16);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_133: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_770, [1, 12, 2048, 2048]);  convert_element_type_770 = None
        view_744: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_133, [12, 2048, 2048]);  expand_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_736: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_114, [2048, 768]);  mul_114 = None
        permute_338: "f16[768, 768]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        mm_174: "f16[2048, 768]" = torch.ops.aten.mm.default(view_736, permute_338);  view_736 = permute_338 = None
        view_737: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_174, [1, 2048, 768]);  mm_174 = None
        view_738: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_737, [1, 2048, -1, 64]);  view_737 = None
        permute_339: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_134: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_339, [1, 12, 2048, 64]);  permute_339 = None
        view_745: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_134, [12, 2048, 64]);  expand_134 = None
        bmm_65: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_744, view_745);  view_744 = view_745 = None
        view_746: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_65, [1, 12, 2048, 64]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_341: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None
        clone_144: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_747: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_144, [1, 2048, -1]);  clone_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_748: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_747, [2048, 768]);  view_747 = None
        permute_342: "f16[768, 768]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        mm_175: "f16[2048, 768]" = torch.ops.aten.mm.default(view_748, permute_342);  view_748 = permute_342 = None
        view_749: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_175, [1, 2048, 768]);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_159: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_758, view_749);  convert_element_type_758 = view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_775: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_159, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_54: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_159);  add_159 = None
        any_55: "b8[]" = torch.ops.aten.any.default(isinf_54);  isinf_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_119: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_59: "f32[]" = torch.ops.aten.where.self(any_55, full_default_119, full_default_118);  any_55 = full_default_119 = full_default_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_55: "f32[]" = torch.ops.aten.neg.default(where_59)
        clamp_min_54: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_775, neg_55);  convert_element_type_775 = neg_55 = None
        clamp_max_54: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_54, where_59);  clamp_min_54 = where_59 = None
        convert_element_type_776: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_54, torch.float16);  clamp_max_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_777: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_776, torch.float32)
        pow_57: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_777, 2);  convert_element_type_777 = None
        mean_56: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_160: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_56, 1e-06);  mean_56 = None
        rsqrt_56: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_115: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_776, rsqrt_56);  rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_778: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_115, torch.float16);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_116: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg236_1, convert_element_type_778);  arg236_1 = convert_element_type_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_750: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_116, [2048, 768]);  mul_116 = None
        permute_343: "f16[768, 768]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        mm_176: "f16[2048, 768]" = torch.ops.aten.mm.default(view_750, permute_343);  view_750 = permute_343 = None
        view_751: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_176, [1, 2048, 768]);  mm_176 = None
        view_752: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_751, [1, 2048, -1, 64]);  view_751 = None
        permute_344: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_135: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_344, [1, 12, 2048, 64]);  permute_344 = None
        view_759: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_135, [12, 2048, 64]);  expand_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_753: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_345: "f16[768, 768]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        mm_177: "f16[2048, 768]" = torch.ops.aten.mm.default(view_753, permute_345);  view_753 = permute_345 = None
        view_754: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_177, [1, 2048, 768]);  mm_177 = None
        view_755: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_754, [1, 2048, -1, 64]);  view_754 = None
        permute_346: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_755, [0, 2, 1, 3]);  view_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_349: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_346, [0, 1, 3, 2]);  permute_346 = None
        expand_136: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_349, [1, 12, 64, 2048]);  permute_349 = None
        view_760: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_136, [12, 64, 2048]);  expand_136 = None
        bmm_66: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_759, view_760);  view_759 = view_760 = None
        view_761: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_66, [1, 12, 2048, 2048]);  bmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_161: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_761, add_80);  view_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_787: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_161, torch.float32);  add_161 = None
        amax_33: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_787, [-1], True)
        sub_35: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_787, amax_33);  convert_element_type_787 = amax_33 = None
        exp_33: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_34: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_37: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        convert_element_type_788: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_37, torch.float16);  div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_137: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_788, [1, 12, 2048, 2048]);  convert_element_type_788 = None
        view_764: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_137, [12, 2048, 2048]);  expand_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_756: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_347: "f16[768, 768]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_178: "f16[2048, 768]" = torch.ops.aten.mm.default(view_756, permute_347);  view_756 = permute_347 = None
        view_757: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_178, [1, 2048, 768]);  mm_178 = None
        view_758: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_757, [1, 2048, -1, 64]);  view_757 = None
        permute_348: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_138: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_348, [1, 12, 2048, 64]);  permute_348 = None
        view_765: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_138, [12, 2048, 64]);  expand_138 = None
        bmm_67: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_764, view_765);  view_764 = view_765 = None
        view_766: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_67, [1, 12, 2048, 64]);  bmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_350: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
        clone_147: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_767: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_147, [1, 2048, -1]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_768: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_767, [2048, 768]);  view_767 = None
        permute_351: "f16[768, 768]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        mm_179: "f16[2048, 768]" = torch.ops.aten.mm.default(view_768, permute_351);  view_768 = permute_351 = None
        view_769: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_179, [1, 2048, 768]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_162: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_776, view_769);  convert_element_type_776 = view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_793: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_162, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_55: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_162);  add_162 = None
        any_56: "b8[]" = torch.ops.aten.any.default(isinf_55);  isinf_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_121: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_120: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_60: "f32[]" = torch.ops.aten.where.self(any_56, full_default_121, full_default_120);  any_56 = full_default_121 = full_default_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_56: "f32[]" = torch.ops.aten.neg.default(where_60)
        clamp_min_55: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_793, neg_56);  convert_element_type_793 = neg_56 = None
        clamp_max_55: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_55, where_60);  clamp_min_55 = where_60 = None
        convert_element_type_794: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_55, torch.float16);  clamp_max_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_795: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_794, torch.float32)
        pow_58: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_795, 2);  convert_element_type_795 = None
        mean_57: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_163: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_57, 1e-06);  mean_57 = None
        rsqrt_57: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_117: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_794, rsqrt_57);  rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_796: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_117, torch.float16);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_118: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg241_1, convert_element_type_796);  arg241_1 = convert_element_type_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_770: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_118, [2048, 768]);  mul_118 = None
        permute_352: "f16[768, 3072]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        mm_180: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_770, permute_352);  view_770 = permute_352 = None
        view_771: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_180, [1, 2048, 3072]);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_22: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_771);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_772: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_22, [2048, 3072]);  relu_22 = None
        permute_353: "f16[3072, 768]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_181: "f16[2048, 768]" = torch.ops.aten.mm.default(view_772, permute_353);  view_772 = permute_353 = None
        view_773: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_181, [1, 2048, 768]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_164: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_794, view_773);  convert_element_type_794 = view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_801: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_164, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_56: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_164);  add_164 = None
        any_57: "b8[]" = torch.ops.aten.any.default(isinf_56);  isinf_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_123: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_122: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_61: "f32[]" = torch.ops.aten.where.self(any_57, full_default_123, full_default_122);  any_57 = full_default_123 = full_default_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_57: "f32[]" = torch.ops.aten.neg.default(where_61)
        clamp_min_56: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_801, neg_57);  convert_element_type_801 = neg_57 = None
        clamp_max_56: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_56, where_61);  clamp_min_56 = where_61 = None
        convert_element_type_802: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_56, torch.float16);  clamp_max_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_803: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_802, torch.float32)
        pow_59: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_803, 2);  convert_element_type_803 = None
        mean_58: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_165: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_58, 1e-06);  mean_58 = None
        rsqrt_58: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_119: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_802, rsqrt_58);  rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_804: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_119, torch.float16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_120: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg244_1, convert_element_type_804);  arg244_1 = convert_element_type_804 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_774: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_120, [2048, 768])
        permute_354: "f16[768, 768]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        mm_182: "f16[2048, 768]" = torch.ops.aten.mm.default(view_774, permute_354);  view_774 = permute_354 = None
        view_775: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_182, [1, 2048, 768]);  mm_182 = None
        view_776: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_775, [1, 2048, -1, 64]);  view_775 = None
        permute_355: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_139: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_355, [1, 12, 2048, 64]);  permute_355 = None
        view_783: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_139, [12, 2048, 64]);  expand_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_777: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_120, [2048, 768])
        permute_356: "f16[768, 768]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        mm_183: "f16[2048, 768]" = torch.ops.aten.mm.default(view_777, permute_356);  view_777 = permute_356 = None
        view_778: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_183, [1, 2048, 768]);  mm_183 = None
        view_779: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_778, [1, 2048, -1, 64]);  view_778 = None
        permute_357: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_779, [0, 2, 1, 3]);  view_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_360: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_357, [0, 1, 3, 2]);  permute_357 = None
        expand_140: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_360, [1, 12, 64, 2048]);  permute_360 = None
        view_784: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_140, [12, 64, 2048]);  expand_140 = None
        bmm_68: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_783, view_784);  view_783 = view_784 = None
        view_785: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_68, [1, 12, 2048, 2048]);  bmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_166: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_785, add_76);  view_785 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_813: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_166, torch.float32);  add_166 = None
        amax_34: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_813, [-1], True)
        sub_36: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_813, amax_34);  convert_element_type_813 = amax_34 = None
        exp_34: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_35: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_38: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        convert_element_type_814: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_38, torch.float16);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_141: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_814, [1, 12, 2048, 2048]);  convert_element_type_814 = None
        view_788: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_141, [12, 2048, 2048]);  expand_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_780: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_120, [2048, 768]);  mul_120 = None
        permute_358: "f16[768, 768]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_184: "f16[2048, 768]" = torch.ops.aten.mm.default(view_780, permute_358);  view_780 = permute_358 = None
        view_781: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_184, [1, 2048, 768]);  mm_184 = None
        view_782: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_781, [1, 2048, -1, 64]);  view_781 = None
        permute_359: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_142: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_359, [1, 12, 2048, 64]);  permute_359 = None
        view_789: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_142, [12, 2048, 64]);  expand_142 = None
        bmm_69: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_788, view_789);  view_788 = view_789 = None
        view_790: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_69, [1, 12, 2048, 64]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_361: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_790, [0, 2, 1, 3]);  view_790 = None
        clone_152: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_791: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_152, [1, 2048, -1]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_792: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_791, [2048, 768]);  view_791 = None
        permute_362: "f16[768, 768]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_185: "f16[2048, 768]" = torch.ops.aten.mm.default(view_792, permute_362);  view_792 = permute_362 = None
        view_793: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_185, [1, 2048, 768]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_167: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_802, view_793);  convert_element_type_802 = view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_819: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_167, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_57: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_167);  add_167 = None
        any_58: "b8[]" = torch.ops.aten.any.default(isinf_57);  isinf_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_125: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_124: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_62: "f32[]" = torch.ops.aten.where.self(any_58, full_default_125, full_default_124);  any_58 = full_default_125 = full_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_58: "f32[]" = torch.ops.aten.neg.default(where_62)
        clamp_min_57: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_819, neg_58);  convert_element_type_819 = neg_58 = None
        clamp_max_57: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_57, where_62);  clamp_min_57 = where_62 = None
        convert_element_type_820: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_57, torch.float16);  clamp_max_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_821: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_820, torch.float32)
        pow_60: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_821, 2);  convert_element_type_821 = None
        mean_59: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_168: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_59, 1e-06);  mean_59 = None
        rsqrt_59: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_121: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_820, rsqrt_59);  rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_822: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_121, torch.float16);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_122: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg249_1, convert_element_type_822);  arg249_1 = convert_element_type_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_794: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_122, [2048, 768]);  mul_122 = None
        permute_363: "f16[768, 768]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_186: "f16[2048, 768]" = torch.ops.aten.mm.default(view_794, permute_363);  view_794 = permute_363 = None
        view_795: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_186, [1, 2048, 768]);  mm_186 = None
        view_796: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_795, [1, 2048, -1, 64]);  view_795 = None
        permute_364: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_796, [0, 2, 1, 3]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_143: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_364, [1, 12, 2048, 64]);  permute_364 = None
        view_803: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_143, [12, 2048, 64]);  expand_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_797: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_365: "f16[768, 768]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        mm_187: "f16[2048, 768]" = torch.ops.aten.mm.default(view_797, permute_365);  view_797 = permute_365 = None
        view_798: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_187, [1, 2048, 768]);  mm_187 = None
        view_799: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_798, [1, 2048, -1, 64]);  view_798 = None
        permute_366: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_369: "f16[1, 12, 64, 2048]" = torch.ops.aten.permute.default(permute_366, [0, 1, 3, 2]);  permute_366 = None
        expand_144: "f16[1, 12, 64, 2048]" = torch.ops.aten.expand.default(permute_369, [1, 12, 64, 2048]);  permute_369 = None
        view_804: "f16[12, 64, 2048]" = torch.ops.aten.reshape.default(expand_144, [12, 64, 2048]);  expand_144 = None
        bmm_70: "f16[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_803, view_804);  view_803 = view_804 = None
        view_805: "f16[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_70, [1, 12, 2048, 2048]);  bmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_169: "f16[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_805, add_80);  view_805 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_831: "f32[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_169, torch.float32);  add_169 = None
        amax_35: "f32[1, 12, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_831, [-1], True)
        sub_37: "f32[1, 12, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_831, amax_35);  convert_element_type_831 = amax_35 = None
        exp_35: "f32[1, 12, 2048, 2048]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_36: "f32[1, 12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_39: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        convert_element_type_832: "f16[1, 12, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_39, torch.float16);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_145: "f16[1, 12, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_832, [1, 12, 2048, 2048]);  convert_element_type_832 = None
        view_808: "f16[12, 2048, 2048]" = torch.ops.aten.reshape.default(expand_145, [12, 2048, 2048]);  expand_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_800: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, [2048, 768])
        permute_367: "f16[768, 768]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        mm_188: "f16[2048, 768]" = torch.ops.aten.mm.default(view_800, permute_367);  view_800 = permute_367 = None
        view_801: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_188, [1, 2048, 768]);  mm_188 = None
        view_802: "f16[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_801, [1, 2048, -1, 64]);  view_801 = None
        permute_368: "f16[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_146: "f16[1, 12, 2048, 64]" = torch.ops.aten.expand.default(permute_368, [1, 12, 2048, 64]);  permute_368 = None
        view_809: "f16[12, 2048, 64]" = torch.ops.aten.reshape.default(expand_146, [12, 2048, 64]);  expand_146 = None
        bmm_71: "f16[12, 2048, 64]" = torch.ops.aten.bmm.default(view_808, view_809);  view_808 = view_809 = None
        view_810: "f16[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_71, [1, 12, 2048, 64]);  bmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_370: "f16[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_810, [0, 2, 1, 3]);  view_810 = None
        clone_155: "f16[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_811: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_155, [1, 2048, -1]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_812: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_811, [2048, 768]);  view_811 = None
        permute_371: "f16[768, 768]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        mm_189: "f16[2048, 768]" = torch.ops.aten.mm.default(view_812, permute_371);  view_812 = permute_371 = None
        view_813: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_189, [1, 2048, 768]);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_170: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_820, view_813);  convert_element_type_820 = view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_837: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_58: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_170);  add_170 = None
        any_59: "b8[]" = torch.ops.aten.any.default(isinf_58);  isinf_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_127: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_126: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_63: "f32[]" = torch.ops.aten.where.self(any_59, full_default_127, full_default_126);  any_59 = full_default_127 = full_default_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_59: "f32[]" = torch.ops.aten.neg.default(where_63)
        clamp_min_58: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_837, neg_59);  convert_element_type_837 = neg_59 = None
        clamp_max_58: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_58, where_63);  clamp_min_58 = where_63 = None
        convert_element_type_838: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_58, torch.float16);  clamp_max_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_839: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_838, torch.float32)
        pow_61: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_839, 2);  convert_element_type_839 = None
        mean_60: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_171: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_60, 1e-06);  mean_60 = None
        rsqrt_60: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_123: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_838, rsqrt_60);  rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_840: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_123, torch.float16);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_124: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg254_1, convert_element_type_840);  arg254_1 = convert_element_type_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_814: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_124, [2048, 768]);  mul_124 = None
        permute_372: "f16[768, 3072]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        mm_190: "f16[2048, 3072]" = torch.ops.aten.mm.default(view_814, permute_372);  view_814 = permute_372 = None
        view_815: "f16[1, 2048, 3072]" = torch.ops.aten.reshape.default(mm_190, [1, 2048, 3072]);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_23: "f16[1, 2048, 3072]" = torch.ops.aten.relu.default(view_815);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_816: "f16[2048, 3072]" = torch.ops.aten.reshape.default(relu_23, [2048, 3072]);  relu_23 = None
        permute_373: "f16[3072, 768]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        mm_191: "f16[2048, 768]" = torch.ops.aten.mm.default(view_816, permute_373);  view_816 = permute_373 = None
        view_817: "f16[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_191, [1, 2048, 768]);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_172: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_838, view_817);  convert_element_type_838 = view_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_845: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_172, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_59: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_172);  add_172 = None
        any_60: "b8[]" = torch.ops.aten.any.default(isinf_59);  isinf_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_129: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_128: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_64: "f32[]" = torch.ops.aten.where.self(any_60, full_default_129, full_default_128);  any_60 = full_default_129 = full_default_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_60: "f32[]" = torch.ops.aten.neg.default(where_64)
        clamp_min_59: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_845, neg_60);  convert_element_type_845 = neg_60 = None
        clamp_max_59: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_59, where_64);  clamp_min_59 = where_64 = None
        convert_element_type_846: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_59, torch.float16);  clamp_max_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_847: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_846, torch.float32)
        pow_62: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_847, 2);  convert_element_type_847 = None
        mean_61: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_173: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_61, 1e-06);  mean_61 = None
        rsqrt_61: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_125: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_846, rsqrt_61);  convert_element_type_846 = rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_848: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_125, torch.float16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_126: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg257_1, convert_element_type_848);  arg257_1 = convert_element_type_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_127: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_126, 0.03608439182435161);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_818: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_127, [2048, 768]);  mul_127 = None
        permute_374: "f16[768, 32128]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_192: "f16[2048, 32128]" = torch.ops.aten.mm.default(view_818, permute_374);  view_818 = permute_374 = None
        view_819: "f16[1, 2048, 32128]" = torch.ops.aten.reshape.default(mm_192, [1, 2048, 32128]);  mm_192 = None
        return (view_819, mul_51)
