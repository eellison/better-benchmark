import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 128]", arg1_1: "f32[250112, 512]", arg2_1: "f32[512]", arg3_1: "f32[384, 512]", arg4_1: "f32[384, 512]", arg5_1: "f32[384, 512]", arg6_1: "f32[32, 6]", arg7_1: "f32[512, 384]", arg8_1: "f32[512]", arg9_1: "f32[1024, 512]", arg10_1: "f32[1024, 512]", arg11_1: "f32[512, 1024]", arg12_1: "f32[512]", arg13_1: "f32[384, 512]", arg14_1: "f32[384, 512]", arg15_1: "f32[384, 512]", arg16_1: "f32[512, 384]", arg17_1: "f32[512]", arg18_1: "f32[1024, 512]", arg19_1: "f32[1024, 512]", arg20_1: "f32[512, 1024]", arg21_1: "f32[512]", arg22_1: "f32[384, 512]", arg23_1: "f32[384, 512]", arg24_1: "f32[384, 512]", arg25_1: "f32[512, 384]", arg26_1: "f32[512]", arg27_1: "f32[1024, 512]", arg28_1: "f32[1024, 512]", arg29_1: "f32[512, 1024]", arg30_1: "f32[512]", arg31_1: "f32[384, 512]", arg32_1: "f32[384, 512]", arg33_1: "f32[384, 512]", arg34_1: "f32[512, 384]", arg35_1: "f32[512]", arg36_1: "f32[1024, 512]", arg37_1: "f32[1024, 512]", arg38_1: "f32[512, 1024]", arg39_1: "f32[512]", arg40_1: "f32[384, 512]", arg41_1: "f32[384, 512]", arg42_1: "f32[384, 512]", arg43_1: "f32[512, 384]", arg44_1: "f32[512]", arg45_1: "f32[1024, 512]", arg46_1: "f32[1024, 512]", arg47_1: "f32[512, 1024]", arg48_1: "f32[512]", arg49_1: "f32[384, 512]", arg50_1: "f32[384, 512]", arg51_1: "f32[384, 512]", arg52_1: "f32[512, 384]", arg53_1: "f32[512]", arg54_1: "f32[1024, 512]", arg55_1: "f32[1024, 512]", arg56_1: "f32[512, 1024]", arg57_1: "f32[512]", arg58_1: "f32[384, 512]", arg59_1: "f32[384, 512]", arg60_1: "f32[384, 512]", arg61_1: "f32[512, 384]", arg62_1: "f32[512]", arg63_1: "f32[1024, 512]", arg64_1: "f32[1024, 512]", arg65_1: "f32[512, 1024]", arg66_1: "f32[512]", arg67_1: "f32[384, 512]", arg68_1: "f32[384, 512]", arg69_1: "f32[384, 512]", arg70_1: "f32[512, 384]", arg71_1: "f32[512]", arg72_1: "f32[1024, 512]", arg73_1: "f32[1024, 512]", arg74_1: "f32[512, 1024]", arg75_1: "f32[512]", arg76_1: "i64[32, 128]", arg77_1: "f32[512]", arg78_1: "f32[384, 512]", arg79_1: "f32[384, 512]", arg80_1: "f32[384, 512]", arg81_1: "f32[32, 6]", arg82_1: "f32[512, 384]", arg83_1: "f32[512]", arg84_1: "f32[384, 512]", arg85_1: "f32[384, 512]", arg86_1: "f32[384, 512]", arg87_1: "f32[512, 384]", arg88_1: "f32[512]", arg89_1: "f32[1024, 512]", arg90_1: "f32[1024, 512]", arg91_1: "f32[512, 1024]", arg92_1: "f32[512]", arg93_1: "f32[384, 512]", arg94_1: "f32[384, 512]", arg95_1: "f32[384, 512]", arg96_1: "f32[512, 384]", arg97_1: "f32[512]", arg98_1: "f32[384, 512]", arg99_1: "f32[384, 512]", arg100_1: "f32[384, 512]", arg101_1: "f32[512, 384]", arg102_1: "f32[512]", arg103_1: "f32[1024, 512]", arg104_1: "f32[1024, 512]", arg105_1: "f32[512, 1024]", arg106_1: "f32[512]", arg107_1: "f32[384, 512]", arg108_1: "f32[384, 512]", arg109_1: "f32[384, 512]", arg110_1: "f32[512, 384]", arg111_1: "f32[512]", arg112_1: "f32[384, 512]", arg113_1: "f32[384, 512]", arg114_1: "f32[384, 512]", arg115_1: "f32[512, 384]", arg116_1: "f32[512]", arg117_1: "f32[1024, 512]", arg118_1: "f32[1024, 512]", arg119_1: "f32[512, 1024]", arg120_1: "f32[512]", arg121_1: "f32[384, 512]", arg122_1: "f32[384, 512]", arg123_1: "f32[384, 512]", arg124_1: "f32[512, 384]", arg125_1: "f32[512]", arg126_1: "f32[384, 512]", arg127_1: "f32[384, 512]", arg128_1: "f32[384, 512]", arg129_1: "f32[512, 384]", arg130_1: "f32[512]", arg131_1: "f32[1024, 512]", arg132_1: "f32[1024, 512]", arg133_1: "f32[512, 1024]", arg134_1: "f32[512]", arg135_1: "f32[384, 512]", arg136_1: "f32[384, 512]", arg137_1: "f32[384, 512]", arg138_1: "f32[512, 384]", arg139_1: "f32[512]", arg140_1: "f32[384, 512]", arg141_1: "f32[384, 512]", arg142_1: "f32[384, 512]", arg143_1: "f32[512, 384]", arg144_1: "f32[512]", arg145_1: "f32[1024, 512]", arg146_1: "f32[1024, 512]", arg147_1: "f32[512, 1024]", arg148_1: "f32[512]", arg149_1: "f32[384, 512]", arg150_1: "f32[384, 512]", arg151_1: "f32[384, 512]", arg152_1: "f32[512, 384]", arg153_1: "f32[512]", arg154_1: "f32[384, 512]", arg155_1: "f32[384, 512]", arg156_1: "f32[384, 512]", arg157_1: "f32[512, 384]", arg158_1: "f32[512]", arg159_1: "f32[1024, 512]", arg160_1: "f32[1024, 512]", arg161_1: "f32[512, 1024]", arg162_1: "f32[512]", arg163_1: "f32[384, 512]", arg164_1: "f32[384, 512]", arg165_1: "f32[384, 512]", arg166_1: "f32[512, 384]", arg167_1: "f32[512]", arg168_1: "f32[384, 512]", arg169_1: "f32[384, 512]", arg170_1: "f32[384, 512]", arg171_1: "f32[512, 384]", arg172_1: "f32[512]", arg173_1: "f32[1024, 512]", arg174_1: "f32[1024, 512]", arg175_1: "f32[512, 1024]", arg176_1: "f32[512]", arg177_1: "f32[384, 512]", arg178_1: "f32[384, 512]", arg179_1: "f32[384, 512]", arg180_1: "f32[512, 384]", arg181_1: "f32[512]", arg182_1: "f32[384, 512]", arg183_1: "f32[384, 512]", arg184_1: "f32[384, 512]", arg185_1: "f32[512, 384]", arg186_1: "f32[512]", arg187_1: "f32[1024, 512]", arg188_1: "f32[1024, 512]", arg189_1: "f32[512, 1024]", arg190_1: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None
        _tensor_constant1: "f32[]" = self._tensor_constant1;  _tensor_constant1 = None
        _tensor_constant2: "f32[]" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "f32[32, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_26: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding_2, 2)
        mean_17: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_68: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_76: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_17);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_77: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg77_1, mul_76);  arg77_1 = mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_77, [4096, 512])
        permute_97: "f32[512, 384]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_56: "f32[4096, 384]" = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
        view_211: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_56, [32, 128, 384]);  mm_56 = None
        view_212: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_211, [32, 128, -1, 64]);  view_211 = None

        # No stacktrace found for following nodes
        permute_default_52: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        mul_scalar_14: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_52, 1.0);  permute_default_52 = None
        expand_default_36: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_14, [32, 6, 128, 64]);  mul_scalar_14 = None
        clone_default_21: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_36, memory_format = torch.contiguous_format);  expand_default_36 = None
        view_default_42: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_21, [192, 128, 64]);  clone_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_213: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_77, [4096, 512])
        permute_99: "f32[512, 384]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_57: "f32[4096, 384]" = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
        view_214: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_57, [32, 128, 384]);  mm_57 = None
        view_215: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_214, [32, 128, -1, 64]);  view_214 = None

        # No stacktrace found for following nodes
        permute_default_53: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        permute_default_55: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_53, [0, 1, 3, 2]);  permute_default_53 = None
        mul_scalar_15: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_55, 1.0);  permute_default_55 = None
        expand_default_37: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_15, [32, 6, 64, 128]);  mul_scalar_15 = None
        clone_default_22: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_37, memory_format = torch.contiguous_format);  expand_default_37 = None
        view_default_43: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_22, [192, 64, 128]);  clone_default_22 = None
        bmm_default_14: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_42, view_default_43);  view_default_42 = view_default_43 = None
        view_default_44: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_14, [32, 6, 128, 128]);  bmm_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:233 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_15: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_16: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_15, 0);  iota_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:232 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_14: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_14, 1);  iota_14 = None
        add_69: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze_15, 0);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:234 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_9: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_16, add_69);  unsqueeze_16 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:208 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_7: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[128, 128]" = torch.ops.aten.minimum.default(sub_9, full_default_7);  sub_9 = full_default_7 = None
        neg: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_3: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_10: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_3, 16);  convert_element_type_3 = None
        log_1: "f32[128, 128]" = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_78: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_4: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_78, torch.int64);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_70: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_4, 16);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_8: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[128, 128]" = torch.ops.aten.minimum.default(add_70, full_default_8);  add_70 = full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_4: "i64[128, 128]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_71: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_4, 0);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg81_1, add_71);  arg81_1 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_104: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_65: "i64[128]" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_65, 0);  add_65 = None
        unsqueeze_10: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_64: "i64[128]" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_64, 0);  add_64 = None
        unsqueeze_7: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_33: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le, [32, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_33, full_default_3, full_default_4);  expand_33 = full_default_3 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_72: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_17, where_2);  unsqueeze_17 = where_2 = None

        # No stacktrace found for following nodes
        add_tensor_7: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_44, add_72);  view_default_44 = None
        eq_scalar_7: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_7, -inf)
        logical_not_default_14: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_7);  eq_scalar_7 = None
        any_dim_7: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_14, -1, True);  logical_not_default_14 = None
        logical_not_default_15: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_7);  any_dim_7 = None
        full_default_19: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_7: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_7, [-1], True)
        sub_tensor_7: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_7, amax_default_7);  add_tensor_7 = amax_default_7 = None
        exp_default_7: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_7);  sub_tensor_7 = None
        sum_dim_int_list_7: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_7, [-1], True)
        div_tensor_7: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_7, sum_dim_int_list_7);  exp_default_7 = sum_dim_int_list_7 = None
        where_self_7: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_15, full_default_19, div_tensor_7);  logical_not_default_15 = full_default_19 = div_tensor_7 = None
        expand_default_38: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_7, [32, 6, 128, 128]);  where_self_7 = None
        view_default_45: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_38, [192, 128, 128]);  expand_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_216: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_77, [4096, 512]);  mul_77 = None
        permute_101: "f32[512, 384]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        mm_58: "f32[4096, 384]" = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
        view_217: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_58, [32, 128, 384]);  mm_58 = None
        view_218: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_217, [32, 128, -1, 64]);  view_217 = None

        # No stacktrace found for following nodes
        permute_default_54: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        expand_default_39: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_54, [32, 6, 128, 64]);  permute_default_54 = None
        clone_default_23: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_39, memory_format = torch.contiguous_format);  expand_default_39 = None
        view_default_46: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_23, [192, 128, 64]);  clone_default_23 = None
        bmm_default_15: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_45, view_default_46);  view_default_45 = view_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_47: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_15, [32, 6, 128, 64]);  bmm_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_105: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_47, [0, 2, 1, 3]);  view_default_47 = None
        clone_71: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_227: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_71, [32, 128, -1]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_228: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_227, [4096, 384]);  view_227 = None
        permute_106: "f32[384, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_59: "f32[4096, 512]" = torch.ops.aten.mm.default(view_228, permute_106);  view_228 = permute_106 = None
        view_229: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_59, [32, 128, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_74: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(embedding_2, view_229);  embedding_2 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_27: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_18: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_75: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_79: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_18);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_80: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg83_1, mul_79);  arg83_1 = mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_230: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_80, [4096, 512]);  mul_80 = None
        permute_107: "f32[512, 384]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_60: "f32[4096, 384]" = torch.ops.aten.mm.default(view_230, permute_107);  view_230 = permute_107 = None
        view_231: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_60, [32, 128, 384]);  mm_60 = None
        view_232: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_231, [32, 128, -1, 64]);  view_231 = None

        # No stacktrace found for following nodes
        permute_default_49: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f32[32, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding, 2)
        mean: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul);  arg2_1 = mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_1, [4096, 512])
        permute: "f32[512, 384]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm, [32, 128, 384]);  mm = None
        view_3: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_2, [32, 128, -1, 64]);  view_2 = None

        # No stacktrace found for following nodes
        permute_default_84: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        mul_scalar_30: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_84, 1.0);  permute_default_84 = None
        expand_default_68: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_30, [32, 6, 128, 64]);  mul_scalar_30 = None
        clone_default_45: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_68, memory_format = torch.contiguous_format);  expand_default_68 = None
        view_default_90: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_45, [192, 128, 64]);  clone_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_4: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_1, [4096, 512])
        permute_2: "f32[512, 384]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f32[4096, 384]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_1, [32, 128, 384]);  mm_1 = None
        view_6: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_5, [32, 128, -1, 64]);  view_5 = None

        # No stacktrace found for following nodes
        permute_default_85: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_default_87: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_85, [0, 1, 3, 2]);  permute_default_85 = None
        mul_scalar_31: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_87, 1.0);  permute_default_87 = None
        expand_default_69: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_31, [32, 6, 64, 128]);  mul_scalar_31 = None
        clone_default_46: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_69, memory_format = torch.contiguous_format);  expand_default_69 = None
        view_default_91: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_46, [192, 64, 128]);  clone_default_46 = None
        bmm_default_30: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_90, view_default_91);  view_default_90 = view_default_91 = None
        view_default_92: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_30, [32, 6, 128, 128]);  bmm_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:233 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_5: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:232 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_4, 1);  iota_4 = None
        add_3: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:234 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:205 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_2: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
        add_4: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:206 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[128, 128]" = torch.ops.aten.abs.default(sub);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_1: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
        log: "f32[128, 128]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_3: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_2: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_2: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[128, 128]" = torch.ops.aten.minimum.default(add_5, full_default_2);  add_5 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[128, 128]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg6_1, add_6);  arg6_1 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(ge, [32, -1, 128, 128]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = where = None

        # No stacktrace found for following nodes
        add_tensor_15: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_92, add_7);  view_default_92 = None
        eq_scalar_15: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_15, -inf)
        logical_not_default_30: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_15);  eq_scalar_15 = None
        any_dim_15: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_30, -1, True);  logical_not_default_30 = None
        logical_not_default_31: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_15);  any_dim_15 = None
        full_default_27: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_15: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_15, [-1], True)
        sub_tensor_15: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_15, amax_default_15);  add_tensor_15 = amax_default_15 = None
        exp_default_15: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_15);  sub_tensor_15 = None
        sum_dim_int_list_15: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_15, [-1], True)
        div_tensor_15: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_15, sum_dim_int_list_15);  exp_default_15 = sum_dim_int_list_15 = None
        where_self_15: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_31, full_default_27, div_tensor_15);  logical_not_default_31 = full_default_27 = div_tensor_15 = None
        expand_default_70: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_15, [32, 6, 128, 128]);  where_self_15 = None
        view_default_93: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_70, [192, 128, 128]);  expand_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_7: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_1, [4096, 512]);  mul_1 = None
        permute_4: "f32[512, 384]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "f32[4096, 384]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_2, [32, 128, 384]);  mm_2 = None
        view_9: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_8, [32, 128, -1, 64]);  view_8 = None

        # No stacktrace found for following nodes
        permute_default_86: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        expand_default_71: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_86, [32, 6, 128, 64]);  permute_default_86 = None
        clone_default_47: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_71, memory_format = torch.contiguous_format);  expand_default_71 = None
        view_default_94: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_47, [192, 128, 64]);  clone_default_47 = None
        bmm_default_31: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_93, view_default_94);  view_default_93 = view_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_95: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_31, [32, 6, 128, 64]);  bmm_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_95, [0, 2, 1, 3]);  view_default_95 = None
        clone_5: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_18: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_5, [32, 128, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_19: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_18, [4096, 384]);  view_18 = None
        permute_9: "f32[384, 512]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "f32[4096, 512]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_3, [32, 128, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_9: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_1: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg8_1, mul_4);  arg8_1 = mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_21: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_5, [4096, 512])
        permute_10: "f32[512, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [32, 128, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_3: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_11: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_22, mul_7);  view_22 = mul_7 = None
        mul_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_11, 0.7978845608028654);  add_11 = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_12: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_23: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_5, [4096, 512]);  mul_5 = None
        permute_11: "f32[512, 1024]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [32, 128, 1024]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, view_24);  mul_9 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_25: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_10, [4096, 1024]);  mul_10 = None
        permute_12: "f32[1024, 512]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_6: "f32[4096, 512]" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_6, [32, 128, 512]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_13: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_9, view_26);  add_9 = view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_4: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
        mean_2: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_11: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_2);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_12: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg12_1, mul_11);  arg12_1 = mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_27: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_12, [4096, 512])
        permute_13: "f32[512, 384]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "f32[4096, 384]" = torch.ops.aten.mm.default(view_27, permute_13);  view_27 = permute_13 = None
        view_28: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_7, [32, 128, 384]);  mm_7 = None
        view_29: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_28, [32, 128, -1, 64]);  view_28 = None

        # No stacktrace found for following nodes
        permute_default_80: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        mul_scalar_28: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_80, 1.0);  permute_default_80 = None
        expand_default_64: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_28, [32, 6, 128, 64]);  mul_scalar_28 = None
        clone_default_42: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_64, memory_format = torch.contiguous_format);  expand_default_64 = None
        view_default_84: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_42, [192, 128, 64]);  clone_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_30: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_12, [4096, 512])
        permute_15: "f32[512, 384]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "f32[4096, 384]" = torch.ops.aten.mm.default(view_30, permute_15);  view_30 = permute_15 = None
        view_31: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_8, [32, 128, 384]);  mm_8 = None
        view_32: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_31, [32, 128, -1, 64]);  view_31 = None

        # No stacktrace found for following nodes
        permute_default_81: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        permute_default_83: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_81, [0, 1, 3, 2]);  permute_default_81 = None
        mul_scalar_29: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_83, 1.0);  permute_default_83 = None
        expand_default_65: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_29, [32, 6, 64, 128]);  mul_scalar_29 = None
        clone_default_43: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_65, memory_format = torch.contiguous_format);  expand_default_65 = None
        view_default_85: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_43, [192, 64, 128]);  clone_default_43 = None
        bmm_default_28: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_84, view_default_85);  view_default_84 = view_default_85 = None
        view_default_86: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_28, [32, 6, 128, 128]);  bmm_default_28 = None
        add_tensor_14: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_86, add_7);  view_default_86 = None
        eq_scalar_14: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_14, -inf)
        logical_not_default_28: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_14);  eq_scalar_14 = None
        any_dim_14: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_28, -1, True);  logical_not_default_28 = None
        logical_not_default_29: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_14);  any_dim_14 = None
        full_default_26: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_14: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_14, [-1], True)
        sub_tensor_14: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_14, amax_default_14);  add_tensor_14 = amax_default_14 = None
        exp_default_14: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_14);  sub_tensor_14 = None
        sum_dim_int_list_14: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_14, [-1], True)
        div_tensor_14: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_14, sum_dim_int_list_14);  exp_default_14 = sum_dim_int_list_14 = None
        where_self_14: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_29, full_default_26, div_tensor_14);  logical_not_default_29 = full_default_26 = div_tensor_14 = None
        expand_default_66: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_14, [32, 6, 128, 128]);  where_self_14 = None
        view_default_87: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_66, [192, 128, 128]);  expand_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_33: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_12, [4096, 512]);  mul_12 = None
        permute_17: "f32[512, 384]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "f32[4096, 384]" = torch.ops.aten.mm.default(view_33, permute_17);  view_33 = permute_17 = None
        view_34: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_9, [32, 128, 384]);  mm_9 = None
        view_35: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_34, [32, 128, -1, 64]);  view_34 = None

        # No stacktrace found for following nodes
        permute_default_82: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        expand_default_67: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_82, [32, 6, 128, 64]);  permute_default_82 = None
        clone_default_44: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_67, memory_format = torch.contiguous_format);  expand_default_67 = None
        view_default_88: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_44, [192, 128, 64]);  clone_default_44 = None
        bmm_default_29: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_87, view_default_88);  view_default_87 = view_default_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_89: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_29, [32, 6, 128, 64]);  bmm_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_20: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_89, [0, 2, 1, 3]);  view_default_89 = None
        clone_13: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_44: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_13, [32, 128, -1]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_45: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_44, [4096, 384]);  view_44 = None
        permute_21: "f32[384, 512]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_10: "f32[4096, 512]" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_10, [32, 128, 512]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_16: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_13, view_46);  add_13 = view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_5: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_3: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_13: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_3);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_14: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg17_1, mul_13);  arg17_1 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_47: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_14, [4096, 512])
        permute_22: "f32[512, 1024]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, [32, 128, 1024]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
        pow_6: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_48, 3.0)
        mul_16: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_18: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_48, mul_16);  view_48 = mul_16 = None
        mul_17: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_18, 0.7978845608028654);  add_18 = None
        tanh_1: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_17);  mul_17 = None
        add_19: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_18: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, add_19);  mul_15 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_49: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_14, [4096, 512]);  mul_14 = None
        permute_23: "f32[512, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_12: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, [32, 128, 1024]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_19: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_18, view_50);  mul_18 = view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_51: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_19, [4096, 1024]);  mul_19 = None
        permute_24: "f32[1024, 512]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_13: "f32[4096, 512]" = torch.ops.aten.mm.default(view_51, permute_24);  view_51 = permute_24 = None
        view_52: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_13, [32, 128, 512]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_20: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_16, view_52);  add_16 = view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_7: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_20, 2)
        mean_4: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_20: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_20, rsqrt_4);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_21: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg21_1, mul_20);  arg21_1 = mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_53: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_21, [4096, 512])
        permute_25: "f32[512, 384]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "f32[4096, 384]" = torch.ops.aten.mm.default(view_53, permute_25);  view_53 = permute_25 = None
        view_54: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_14, [32, 128, 384]);  mm_14 = None
        view_55: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_54, [32, 128, -1, 64]);  view_54 = None

        # No stacktrace found for following nodes
        permute_default_76: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        mul_scalar_26: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_76, 1.0);  permute_default_76 = None
        expand_default_60: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_26, [32, 6, 128, 64]);  mul_scalar_26 = None
        clone_default_39: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_60, memory_format = torch.contiguous_format);  expand_default_60 = None
        view_default_78: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_39, [192, 128, 64]);  clone_default_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_56: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_21, [4096, 512])
        permute_27: "f32[512, 384]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "f32[4096, 384]" = torch.ops.aten.mm.default(view_56, permute_27);  view_56 = permute_27 = None
        view_57: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_15, [32, 128, 384]);  mm_15 = None
        view_58: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_57, [32, 128, -1, 64]);  view_57 = None

        # No stacktrace found for following nodes
        permute_default_77: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        permute_default_79: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_77, [0, 1, 3, 2]);  permute_default_77 = None
        mul_scalar_27: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_79, 1.0);  permute_default_79 = None
        expand_default_61: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_27, [32, 6, 64, 128]);  mul_scalar_27 = None
        clone_default_40: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_61, memory_format = torch.contiguous_format);  expand_default_61 = None
        view_default_79: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_40, [192, 64, 128]);  clone_default_40 = None
        bmm_default_26: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_78, view_default_79);  view_default_78 = view_default_79 = None
        view_default_80: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_26, [32, 6, 128, 128]);  bmm_default_26 = None
        add_tensor_13: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_80, add_7);  view_default_80 = None
        eq_scalar_13: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_13, -inf)
        logical_not_default_26: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_13);  eq_scalar_13 = None
        any_dim_13: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_26, -1, True);  logical_not_default_26 = None
        logical_not_default_27: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_13);  any_dim_13 = None
        full_default_25: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_13: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_13, [-1], True)
        sub_tensor_13: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_13, amax_default_13);  add_tensor_13 = amax_default_13 = None
        exp_default_13: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_13);  sub_tensor_13 = None
        sum_dim_int_list_13: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_13, [-1], True)
        div_tensor_13: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_13, sum_dim_int_list_13);  exp_default_13 = sum_dim_int_list_13 = None
        where_self_13: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_27, full_default_25, div_tensor_13);  logical_not_default_27 = full_default_25 = div_tensor_13 = None
        expand_default_62: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_13, [32, 6, 128, 128]);  where_self_13 = None
        view_default_81: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_62, [192, 128, 128]);  expand_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_59: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_21, [4096, 512]);  mul_21 = None
        permute_29: "f32[512, 384]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_16: "f32[4096, 384]" = torch.ops.aten.mm.default(view_59, permute_29);  view_59 = permute_29 = None
        view_60: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_16, [32, 128, 384]);  mm_16 = None
        view_61: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_60, [32, 128, -1, 64]);  view_60 = None

        # No stacktrace found for following nodes
        permute_default_78: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        expand_default_63: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_78, [32, 6, 128, 64]);  permute_default_78 = None
        clone_default_41: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_63, memory_format = torch.contiguous_format);  expand_default_63 = None
        view_default_82: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_41, [192, 128, 64]);  clone_default_41 = None
        bmm_default_27: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_81, view_default_82);  view_default_81 = view_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_83: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_27, [32, 6, 128, 64]);  bmm_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_32: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_83, [0, 2, 1, 3]);  view_default_83 = None
        clone_21: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_70: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_21, [32, 128, -1]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_71: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_70, [4096, 384]);  view_70 = None
        permute_33: "f32[384, 512]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_17: "f32[4096, 512]" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_17, [32, 128, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_23: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_20, view_72);  add_20 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_8: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
        mean_5: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_22: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_5);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_23: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg26_1, mul_22);  arg26_1 = mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_73: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_23, [4096, 512])
        permute_34: "f32[512, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_18: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_18, [32, 128, 1024]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_24: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
        pow_9: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_74, 3.0)
        mul_25: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_25: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_74, mul_25);  view_74 = mul_25 = None
        mul_26: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_26);  mul_26 = None
        add_26: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_27: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_24, add_26);  mul_24 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_75: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_23, [4096, 512]);  mul_23 = None
        permute_35: "f32[512, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_19: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_75, permute_35);  view_75 = permute_35 = None
        view_76: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_19, [32, 128, 1024]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_28: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_27, view_76);  mul_27 = view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_77: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_28, [4096, 1024]);  mul_28 = None
        permute_36: "f32[1024, 512]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_20: "f32[4096, 512]" = torch.ops.aten.mm.default(view_77, permute_36);  view_77 = permute_36 = None
        view_78: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_20, [32, 128, 512]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_27: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_23, view_78);  add_23 = view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_10: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_6: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_29: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_6);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_30: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg30_1, mul_29);  arg30_1 = mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_79: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_30, [4096, 512])
        permute_37: "f32[512, 384]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "f32[4096, 384]" = torch.ops.aten.mm.default(view_79, permute_37);  view_79 = permute_37 = None
        view_80: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_21, [32, 128, 384]);  mm_21 = None
        view_81: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_80, [32, 128, -1, 64]);  view_80 = None

        # No stacktrace found for following nodes
        permute_default_72: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        mul_scalar_24: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_72, 1.0);  permute_default_72 = None
        expand_default_56: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_24, [32, 6, 128, 64]);  mul_scalar_24 = None
        clone_default_36: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_56, memory_format = torch.contiguous_format);  expand_default_56 = None
        view_default_72: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_36, [192, 128, 64]);  clone_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_82: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_30, [4096, 512])
        permute_39: "f32[512, 384]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_22: "f32[4096, 384]" = torch.ops.aten.mm.default(view_82, permute_39);  view_82 = permute_39 = None
        view_83: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_22, [32, 128, 384]);  mm_22 = None
        view_84: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_83, [32, 128, -1, 64]);  view_83 = None

        # No stacktrace found for following nodes
        permute_default_73: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        permute_default_75: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_73, [0, 1, 3, 2]);  permute_default_73 = None
        mul_scalar_25: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_75, 1.0);  permute_default_75 = None
        expand_default_57: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_25, [32, 6, 64, 128]);  mul_scalar_25 = None
        clone_default_37: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_57, memory_format = torch.contiguous_format);  expand_default_57 = None
        view_default_73: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_37, [192, 64, 128]);  clone_default_37 = None
        bmm_default_24: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_72, view_default_73);  view_default_72 = view_default_73 = None
        view_default_74: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_24, [32, 6, 128, 128]);  bmm_default_24 = None
        add_tensor_12: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_74, add_7);  view_default_74 = None
        eq_scalar_12: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_12, -inf)
        logical_not_default_24: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_12);  eq_scalar_12 = None
        any_dim_12: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_24, -1, True);  logical_not_default_24 = None
        logical_not_default_25: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_12);  any_dim_12 = None
        full_default_24: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_12: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_12, [-1], True)
        sub_tensor_12: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_12, amax_default_12);  add_tensor_12 = amax_default_12 = None
        exp_default_12: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_12);  sub_tensor_12 = None
        sum_dim_int_list_12: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_12, [-1], True)
        div_tensor_12: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_12, sum_dim_int_list_12);  exp_default_12 = sum_dim_int_list_12 = None
        where_self_12: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_25, full_default_24, div_tensor_12);  logical_not_default_25 = full_default_24 = div_tensor_12 = None
        expand_default_58: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_12, [32, 6, 128, 128]);  where_self_12 = None
        view_default_75: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_58, [192, 128, 128]);  expand_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_85: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_30, [4096, 512]);  mul_30 = None
        permute_41: "f32[512, 384]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_23: "f32[4096, 384]" = torch.ops.aten.mm.default(view_85, permute_41);  view_85 = permute_41 = None
        view_86: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_23, [32, 128, 384]);  mm_23 = None
        view_87: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_86, [32, 128, -1, 64]);  view_86 = None

        # No stacktrace found for following nodes
        permute_default_74: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        expand_default_59: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_74, [32, 6, 128, 64]);  permute_default_74 = None
        clone_default_38: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_59, memory_format = torch.contiguous_format);  expand_default_59 = None
        view_default_76: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_38, [192, 128, 64]);  clone_default_38 = None
        bmm_default_25: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_75, view_default_76);  view_default_75 = view_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_77: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_25, [32, 6, 128, 64]);  bmm_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_44: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_77, [0, 2, 1, 3]);  view_default_77 = None
        clone_29: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_96: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_29, [32, 128, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_97: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_96, [4096, 384]);  view_96 = None
        permute_45: "f32[384, 512]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_24: "f32[4096, 512]" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_24, [32, 128, 512]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_30: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_27, view_98);  add_27 = view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_11: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_30, 2)
        mean_7: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_31: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_30, rsqrt_7);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_32: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg35_1, mul_31);  arg35_1 = mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_99: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_32, [4096, 512])
        permute_46: "f32[512, 1024]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_25: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_99, permute_46);  view_99 = permute_46 = None
        view_100: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_25, [32, 128, 1024]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_33: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_100, 0.5)
        pow_12: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_100, 3.0)
        mul_34: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_32: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_100, mul_34);  view_100 = mul_34 = None
        mul_35: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        tanh_3: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_35);  mul_35 = None
        add_33: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_36: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, add_33);  mul_33 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_101: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_32, [4096, 512]);  mul_32 = None
        permute_47: "f32[512, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_26: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_101, permute_47);  view_101 = permute_47 = None
        view_102: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_26, [32, 128, 1024]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_37: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_36, view_102);  mul_36 = view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_103: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_37, [4096, 1024]);  mul_37 = None
        permute_48: "f32[1024, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_27: "f32[4096, 512]" = torch.ops.aten.mm.default(view_103, permute_48);  view_103 = permute_48 = None
        view_104: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_27, [32, 128, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_34: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_30, view_104);  add_30 = view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_13: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_8: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_38: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_8);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_39: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg39_1, mul_38);  arg39_1 = mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_105: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_39, [4096, 512])
        permute_49: "f32[512, 384]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_28: "f32[4096, 384]" = torch.ops.aten.mm.default(view_105, permute_49);  view_105 = permute_49 = None
        view_106: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_28, [32, 128, 384]);  mm_28 = None
        view_107: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_106, [32, 128, -1, 64]);  view_106 = None

        # No stacktrace found for following nodes
        permute_default_68: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        mul_scalar_22: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_68, 1.0);  permute_default_68 = None
        expand_default_52: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_22, [32, 6, 128, 64]);  mul_scalar_22 = None
        clone_default_33: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_52, memory_format = torch.contiguous_format);  expand_default_52 = None
        view_default_66: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_33, [192, 128, 64]);  clone_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_108: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_39, [4096, 512])
        permute_51: "f32[512, 384]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_29: "f32[4096, 384]" = torch.ops.aten.mm.default(view_108, permute_51);  view_108 = permute_51 = None
        view_109: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_29, [32, 128, 384]);  mm_29 = None
        view_110: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_109, [32, 128, -1, 64]);  view_109 = None

        # No stacktrace found for following nodes
        permute_default_69: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        permute_default_71: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_69, [0, 1, 3, 2]);  permute_default_69 = None
        mul_scalar_23: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_71, 1.0);  permute_default_71 = None
        expand_default_53: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_23, [32, 6, 64, 128]);  mul_scalar_23 = None
        clone_default_34: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_53, memory_format = torch.contiguous_format);  expand_default_53 = None
        view_default_67: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_34, [192, 64, 128]);  clone_default_34 = None
        bmm_default_22: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_66, view_default_67);  view_default_66 = view_default_67 = None
        view_default_68: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_22, [32, 6, 128, 128]);  bmm_default_22 = None
        add_tensor_11: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_68, add_7);  view_default_68 = None
        eq_scalar_11: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_11, -inf)
        logical_not_default_22: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_11);  eq_scalar_11 = None
        any_dim_11: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_22, -1, True);  logical_not_default_22 = None
        logical_not_default_23: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_11);  any_dim_11 = None
        full_default_23: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_11: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_11, [-1], True)
        sub_tensor_11: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_11, amax_default_11);  add_tensor_11 = amax_default_11 = None
        exp_default_11: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_11);  sub_tensor_11 = None
        sum_dim_int_list_11: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_11, [-1], True)
        div_tensor_11: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_11, sum_dim_int_list_11);  exp_default_11 = sum_dim_int_list_11 = None
        where_self_11: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_23, full_default_23, div_tensor_11);  logical_not_default_23 = full_default_23 = div_tensor_11 = None
        expand_default_54: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_11, [32, 6, 128, 128]);  where_self_11 = None
        view_default_69: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_54, [192, 128, 128]);  expand_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_111: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_39, [4096, 512]);  mul_39 = None
        permute_53: "f32[512, 384]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_30: "f32[4096, 384]" = torch.ops.aten.mm.default(view_111, permute_53);  view_111 = permute_53 = None
        view_112: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_30, [32, 128, 384]);  mm_30 = None
        view_113: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_112, [32, 128, -1, 64]);  view_112 = None

        # No stacktrace found for following nodes
        permute_default_70: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        expand_default_55: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_70, [32, 6, 128, 64]);  permute_default_70 = None
        clone_default_35: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_55, memory_format = torch.contiguous_format);  expand_default_55 = None
        view_default_70: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_35, [192, 128, 64]);  clone_default_35 = None
        bmm_default_23: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_69, view_default_70);  view_default_69 = view_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_71: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_23, [32, 6, 128, 64]);  bmm_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_71, [0, 2, 1, 3]);  view_default_71 = None
        clone_37: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_122: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_37, [32, 128, -1]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_123: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_122, [4096, 384]);  view_122 = None
        permute_57: "f32[384, 512]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_31: "f32[4096, 512]" = torch.ops.aten.mm.default(view_123, permute_57);  view_123 = permute_57 = None
        view_124: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_31, [32, 128, 512]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_37: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_34, view_124);  add_34 = view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_14: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_37, 2)
        mean_9: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_40: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_37, rsqrt_9);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_41: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg44_1, mul_40);  arg44_1 = mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_125: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_41, [4096, 512])
        permute_58: "f32[512, 1024]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_32: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_125, permute_58);  view_125 = permute_58 = None
        view_126: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_32, [32, 128, 1024]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_42: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
        pow_15: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_126, 3.0)
        mul_43: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_39: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_126, mul_43);  view_126 = mul_43 = None
        mul_44: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_44);  mul_44 = None
        add_40: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_45: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_42, add_40);  mul_42 = add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_127: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_41, [4096, 512]);  mul_41 = None
        permute_59: "f32[512, 1024]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_33: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_127, permute_59);  view_127 = permute_59 = None
        view_128: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_33, [32, 128, 1024]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_46: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_45, view_128);  mul_45 = view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_129: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_46, [4096, 1024]);  mul_46 = None
        permute_60: "f32[1024, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_34: "f32[4096, 512]" = torch.ops.aten.mm.default(view_129, permute_60);  view_129 = permute_60 = None
        view_130: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_34, [32, 128, 512]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_41: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_37, view_130);  add_37 = view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_16: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 2)
        mean_10: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_47: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_10);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_48: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg48_1, mul_47);  arg48_1 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_131: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_48, [4096, 512])
        permute_61: "f32[512, 384]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_35: "f32[4096, 384]" = torch.ops.aten.mm.default(view_131, permute_61);  view_131 = permute_61 = None
        view_132: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_35, [32, 128, 384]);  mm_35 = None
        view_133: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_132, [32, 128, -1, 64]);  view_132 = None

        # No stacktrace found for following nodes
        permute_default_64: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        mul_scalar_20: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_64, 1.0);  permute_default_64 = None
        expand_default_48: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_20, [32, 6, 128, 64]);  mul_scalar_20 = None
        clone_default_30: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_48, memory_format = torch.contiguous_format);  expand_default_48 = None
        view_default_60: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_30, [192, 128, 64]);  clone_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_134: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_48, [4096, 512])
        permute_63: "f32[512, 384]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_36: "f32[4096, 384]" = torch.ops.aten.mm.default(view_134, permute_63);  view_134 = permute_63 = None
        view_135: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_36, [32, 128, 384]);  mm_36 = None
        view_136: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_135, [32, 128, -1, 64]);  view_135 = None

        # No stacktrace found for following nodes
        permute_default_65: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_default_67: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_65, [0, 1, 3, 2]);  permute_default_65 = None
        mul_scalar_21: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_67, 1.0);  permute_default_67 = None
        expand_default_49: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_21, [32, 6, 64, 128]);  mul_scalar_21 = None
        clone_default_31: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_49, memory_format = torch.contiguous_format);  expand_default_49 = None
        view_default_61: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_31, [192, 64, 128]);  clone_default_31 = None
        bmm_default_20: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_60, view_default_61);  view_default_60 = view_default_61 = None
        view_default_62: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_20, [32, 6, 128, 128]);  bmm_default_20 = None
        add_tensor_10: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_62, add_7);  view_default_62 = None
        eq_scalar_10: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_10, -inf)
        logical_not_default_20: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_10);  eq_scalar_10 = None
        any_dim_10: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_20, -1, True);  logical_not_default_20 = None
        logical_not_default_21: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_10);  any_dim_10 = None
        full_default_22: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_10: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_10, [-1], True)
        sub_tensor_10: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_10, amax_default_10);  add_tensor_10 = amax_default_10 = None
        exp_default_10: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_10);  sub_tensor_10 = None
        sum_dim_int_list_10: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_10, [-1], True)
        div_tensor_10: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_10, sum_dim_int_list_10);  exp_default_10 = sum_dim_int_list_10 = None
        where_self_10: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_21, full_default_22, div_tensor_10);  logical_not_default_21 = full_default_22 = div_tensor_10 = None
        expand_default_50: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_10, [32, 6, 128, 128]);  where_self_10 = None
        view_default_63: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_50, [192, 128, 128]);  expand_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_137: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_48, [4096, 512]);  mul_48 = None
        permute_65: "f32[512, 384]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_37: "f32[4096, 384]" = torch.ops.aten.mm.default(view_137, permute_65);  view_137 = permute_65 = None
        view_138: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_37, [32, 128, 384]);  mm_37 = None
        view_139: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_138, [32, 128, -1, 64]);  view_138 = None

        # No stacktrace found for following nodes
        permute_default_66: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        expand_default_51: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_66, [32, 6, 128, 64]);  permute_default_66 = None
        clone_default_32: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_51, memory_format = torch.contiguous_format);  expand_default_51 = None
        view_default_64: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_32, [192, 128, 64]);  clone_default_32 = None
        bmm_default_21: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_63, view_default_64);  view_default_63 = view_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_65: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_21, [32, 6, 128, 64]);  bmm_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_68: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_65, [0, 2, 1, 3]);  view_default_65 = None
        clone_45: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_148: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_45, [32, 128, -1]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_149: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_148, [4096, 384]);  view_148 = None
        permute_69: "f32[384, 512]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_38: "f32[4096, 512]" = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
        view_150: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_38, [32, 128, 512]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_44: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_41, view_150);  add_41 = view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_17: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
        mean_11: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_49: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_11);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_50: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg53_1, mul_49);  arg53_1 = mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_151: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_50, [4096, 512])
        permute_70: "f32[512, 1024]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_39: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_151, permute_70);  view_151 = permute_70 = None
        view_152: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_39, [32, 128, 1024]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_51: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
        pow_18: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_152, 3.0)
        mul_52: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_46: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_152, mul_52);  view_152 = mul_52 = None
        mul_53: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_53);  mul_53 = None
        add_47: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_54: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_51, add_47);  mul_51 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_153: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_50, [4096, 512]);  mul_50 = None
        permute_71: "f32[512, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_40: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_153, permute_71);  view_153 = permute_71 = None
        view_154: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_40, [32, 128, 1024]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_55: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_54, view_154);  mul_54 = view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_155: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_55, [4096, 1024]);  mul_55 = None
        permute_72: "f32[1024, 512]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_41: "f32[4096, 512]" = torch.ops.aten.mm.default(view_155, permute_72);  view_155 = permute_72 = None
        view_156: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_41, [32, 128, 512]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_48: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_44, view_156);  add_44 = view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_19: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
        mean_12: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_56: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_12);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_57: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg57_1, mul_56);  arg57_1 = mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_157: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_57, [4096, 512])
        permute_73: "f32[512, 384]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_42: "f32[4096, 384]" = torch.ops.aten.mm.default(view_157, permute_73);  view_157 = permute_73 = None
        view_158: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_42, [32, 128, 384]);  mm_42 = None
        view_159: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_158, [32, 128, -1, 64]);  view_158 = None

        # No stacktrace found for following nodes
        permute_default_60: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        mul_scalar_18: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_60, 1.0);  permute_default_60 = None
        expand_default_44: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_18, [32, 6, 128, 64]);  mul_scalar_18 = None
        clone_default_27: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_44, memory_format = torch.contiguous_format);  expand_default_44 = None
        view_default_54: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_27, [192, 128, 64]);  clone_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_160: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_57, [4096, 512])
        permute_75: "f32[512, 384]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_43: "f32[4096, 384]" = torch.ops.aten.mm.default(view_160, permute_75);  view_160 = permute_75 = None
        view_161: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_43, [32, 128, 384]);  mm_43 = None
        view_162: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_161, [32, 128, -1, 64]);  view_161 = None

        # No stacktrace found for following nodes
        permute_default_61: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_default_63: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_61, [0, 1, 3, 2]);  permute_default_61 = None
        mul_scalar_19: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_63, 1.0);  permute_default_63 = None
        expand_default_45: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_19, [32, 6, 64, 128]);  mul_scalar_19 = None
        clone_default_28: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_45, memory_format = torch.contiguous_format);  expand_default_45 = None
        view_default_55: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_28, [192, 64, 128]);  clone_default_28 = None
        bmm_default_18: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_54, view_default_55);  view_default_54 = view_default_55 = None
        view_default_56: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_18, [32, 6, 128, 128]);  bmm_default_18 = None
        add_tensor_9: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_56, add_7);  view_default_56 = None
        eq_scalar_9: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_9, -inf)
        logical_not_default_18: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_9);  eq_scalar_9 = None
        any_dim_9: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_18, -1, True);  logical_not_default_18 = None
        logical_not_default_19: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_9);  any_dim_9 = None
        full_default_21: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_9: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_9, [-1], True)
        sub_tensor_9: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_9, amax_default_9);  add_tensor_9 = amax_default_9 = None
        exp_default_9: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_9);  sub_tensor_9 = None
        sum_dim_int_list_9: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_9, [-1], True)
        div_tensor_9: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_9, sum_dim_int_list_9);  exp_default_9 = sum_dim_int_list_9 = None
        where_self_9: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_19, full_default_21, div_tensor_9);  logical_not_default_19 = full_default_21 = div_tensor_9 = None
        expand_default_46: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_9, [32, 6, 128, 128]);  where_self_9 = None
        view_default_57: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_46, [192, 128, 128]);  expand_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_163: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_57, [4096, 512]);  mul_57 = None
        permute_77: "f32[512, 384]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_44: "f32[4096, 384]" = torch.ops.aten.mm.default(view_163, permute_77);  view_163 = permute_77 = None
        view_164: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_44, [32, 128, 384]);  mm_44 = None
        view_165: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_164, [32, 128, -1, 64]);  view_164 = None

        # No stacktrace found for following nodes
        permute_default_62: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        expand_default_47: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_62, [32, 6, 128, 64]);  permute_default_62 = None
        clone_default_29: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_47, memory_format = torch.contiguous_format);  expand_default_47 = None
        view_default_58: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_29, [192, 128, 64]);  clone_default_29 = None
        bmm_default_19: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_57, view_default_58);  view_default_57 = view_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_59: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_19, [32, 6, 128, 64]);  bmm_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_80: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_59, [0, 2, 1, 3]);  view_default_59 = None
        clone_53: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_174: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_53, [32, 128, -1]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_175: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_174, [4096, 384]);  view_174 = None
        permute_81: "f32[384, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_45: "f32[4096, 512]" = torch.ops.aten.mm.default(view_175, permute_81);  view_175 = permute_81 = None
        view_176: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_45, [32, 128, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_51: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_48, view_176);  add_48 = view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_20: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_51, 2)
        mean_13: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_58: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_13);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_59: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg62_1, mul_58);  arg62_1 = mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_177: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_59, [4096, 512])
        permute_82: "f32[512, 1024]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_46: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_177, permute_82);  view_177 = permute_82 = None
        view_178: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_46, [32, 128, 1024]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        pow_21: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_61: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_53: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_178, mul_61);  view_178 = mul_61 = None
        mul_62: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_6: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_54: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_63: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, add_54);  mul_60 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_179: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_59, [4096, 512]);  mul_59 = None
        permute_83: "f32[512, 1024]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_47: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_179, permute_83);  view_179 = permute_83 = None
        view_180: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_47, [32, 128, 1024]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_64: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_63, view_180);  mul_63 = view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_181: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_64, [4096, 1024]);  mul_64 = None
        permute_84: "f32[1024, 512]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_48: "f32[4096, 512]" = torch.ops.aten.mm.default(view_181, permute_84);  view_181 = permute_84 = None
        view_182: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_48, [32, 128, 512]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_55: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_51, view_182);  add_51 = view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_22: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_55, 2)
        mean_14: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_65: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_14);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_66: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg66_1, mul_65);  arg66_1 = mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_183: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_66, [4096, 512])
        permute_85: "f32[512, 384]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_49: "f32[4096, 384]" = torch.ops.aten.mm.default(view_183, permute_85);  view_183 = permute_85 = None
        view_184: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_49, [32, 128, 384]);  mm_49 = None
        view_185: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_184, [32, 128, -1, 64]);  view_184 = None

        # No stacktrace found for following nodes
        permute_default_56: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        mul_scalar_16: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_56, 1.0);  permute_default_56 = None
        expand_default_40: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_16, [32, 6, 128, 64]);  mul_scalar_16 = None
        clone_default_24: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_40, memory_format = torch.contiguous_format);  expand_default_40 = None
        view_default_48: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_24, [192, 128, 64]);  clone_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_186: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_66, [4096, 512])
        permute_87: "f32[512, 384]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_50: "f32[4096, 384]" = torch.ops.aten.mm.default(view_186, permute_87);  view_186 = permute_87 = None
        view_187: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_50, [32, 128, 384]);  mm_50 = None
        view_188: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_187, [32, 128, -1, 64]);  view_187 = None

        # No stacktrace found for following nodes
        permute_default_57: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        permute_default_59: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_57, [0, 1, 3, 2]);  permute_default_57 = None
        mul_scalar_17: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_59, 1.0);  permute_default_59 = None
        expand_default_41: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_17, [32, 6, 64, 128]);  mul_scalar_17 = None
        clone_default_25: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_41, memory_format = torch.contiguous_format);  expand_default_41 = None
        view_default_49: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_25, [192, 64, 128]);  clone_default_25 = None
        bmm_default_16: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_48, view_default_49);  view_default_48 = view_default_49 = None
        view_default_50: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_16, [32, 6, 128, 128]);  bmm_default_16 = None
        add_tensor_8: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_50, add_7);  view_default_50 = add_7 = None
        eq_scalar_8: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_8, -inf)
        logical_not_default_16: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_8);  eq_scalar_8 = None
        any_dim_8: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_16, -1, True);  logical_not_default_16 = None
        logical_not_default_17: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_8);  any_dim_8 = None
        full_default_20: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_8: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_8, [-1], True)
        sub_tensor_8: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_8, amax_default_8);  add_tensor_8 = amax_default_8 = None
        exp_default_8: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_8);  sub_tensor_8 = None
        sum_dim_int_list_8: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_8, [-1], True)
        div_tensor_8: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_8, sum_dim_int_list_8);  exp_default_8 = sum_dim_int_list_8 = None
        where_self_8: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_17, full_default_20, div_tensor_8);  logical_not_default_17 = full_default_20 = div_tensor_8 = None
        expand_default_42: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_8, [32, 6, 128, 128]);  where_self_8 = None
        view_default_51: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_42, [192, 128, 128]);  expand_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_189: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_66, [4096, 512]);  mul_66 = None
        permute_89: "f32[512, 384]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_51: "f32[4096, 384]" = torch.ops.aten.mm.default(view_189, permute_89);  view_189 = permute_89 = None
        view_190: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_51, [32, 128, 384]);  mm_51 = None
        view_191: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_190, [32, 128, -1, 64]);  view_190 = None

        # No stacktrace found for following nodes
        permute_default_58: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        expand_default_43: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_58, [32, 6, 128, 64]);  permute_default_58 = None
        clone_default_26: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_43, memory_format = torch.contiguous_format);  expand_default_43 = None
        view_default_52: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_26, [192, 128, 64]);  clone_default_26 = None
        bmm_default_17: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_51, view_default_52);  view_default_51 = view_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_53: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_17, [32, 6, 128, 64]);  bmm_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_92: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_53, [0, 2, 1, 3]);  view_default_53 = None
        clone_61: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_200: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_61, [32, 128, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_201: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_200, [4096, 384]);  view_200 = None
        permute_93: "f32[384, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_52: "f32[4096, 512]" = torch.ops.aten.mm.default(view_201, permute_93);  view_201 = permute_93 = None
        view_202: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_52, [32, 128, 512]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_58: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_55, view_202);  add_55 = view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_23: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_58, 2)
        mean_15: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_67: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_58, rsqrt_15);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_68: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg71_1, mul_67);  arg71_1 = mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_203: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_68, [4096, 512])
        permute_94: "f32[512, 1024]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_53: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_203, permute_94);  view_203 = permute_94 = None
        view_204: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, [32, 128, 1024]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_69: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
        pow_24: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_204, 3.0)
        mul_70: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_60: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_204, mul_70);  view_204 = mul_70 = None
        mul_71: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_60, 0.7978845608028654);  add_60 = None
        tanh_7: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
        add_61: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_72: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_69, add_61);  mul_69 = add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_205: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_68, [4096, 512]);  mul_68 = None
        permute_95: "f32[512, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_54: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_205, permute_95);  view_205 = permute_95 = None
        view_206: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_54, [32, 128, 1024]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_73: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_72, view_206);  mul_72 = view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_207: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_73, [4096, 1024]);  mul_73 = None
        permute_96: "f32[1024, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_55: "f32[4096, 512]" = torch.ops.aten.mm.default(view_207, permute_96);  view_207 = permute_96 = None
        view_208: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_55, [32, 128, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_62: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_58, view_208);  add_58 = view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_25: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_16: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_74: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_16);  add_62 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_75: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg75_1, mul_74);  arg75_1 = mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_233: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_109: "f32[512, 384]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_61: "f32[4096, 384]" = torch.ops.aten.mm.default(view_233, permute_109);  view_233 = permute_109 = None
        view_234: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_61, [32, 128, 384]);  mm_61 = None
        view_235: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_234, [32, 128, -1, 64]);  view_234 = None

        # No stacktrace found for following nodes
        permute_default_50: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_236: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_111: "f32[512, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_62: "f32[4096, 384]" = torch.ops.aten.mm.default(view_236, permute_111);  view_236 = permute_111 = None
        view_237: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_62, [32, 128, 384]);  mm_62 = None
        view_238: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_237, [32, 128, -1, 64]);  view_237 = None

        # No stacktrace found for following nodes
        permute_default_51: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:305 in forward, code: position_bias = torch.zeros(
        full_default_9: "f32[1, 6, 128, 128]" = torch.ops.aten.full.default([1, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_66: "i64[128]" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_12: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_66, 0);  add_66 = None
        unsqueeze_13: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        unsqueeze_14: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_14, 0);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_34: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(ge_1, [32, -1, 128, 128]);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_34, full_default_5, full_default_6);  expand_34 = full_default_5 = full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_76: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(full_default_9, where_3);  full_default_9 = where_3 = None

        # No stacktrace found for following nodes
        expand_default_35: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_49, permute_default_50, permute_default_51, expand_default_35, False, scale = 1.0);  permute_default_49 = permute_default_50 = permute_default_51 = expand_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_7: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_114: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3]);  getitem_7 = None
        clone_77: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_247: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_77, [32, 128, -1]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_248: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_247, [4096, 384]);  view_247 = None
        permute_115: "f32[384, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_63: "f32[4096, 512]" = torch.ops.aten.mm.default(view_248, permute_115);  view_248 = permute_115 = None
        view_249: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_63, [32, 128, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_78: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_74, view_249);  add_74 = view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_28: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_19: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_81: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_19);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_82: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg88_1, mul_81);  arg88_1 = mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_250: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_82, [4096, 512])
        permute_116: "f32[512, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_64: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_250, permute_116);  view_250 = permute_116 = None
        view_251: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_64, [32, 128, 1024]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_83: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
        pow_29: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
        mul_84: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_80: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_251, mul_84);  view_251 = mul_84 = None
        mul_85: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_85);  mul_85 = None
        add_81: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_86: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_83, add_81);  mul_83 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_252: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_82, [4096, 512]);  mul_82 = None
        permute_117: "f32[512, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_65: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_252, permute_117);  view_252 = permute_117 = None
        view_253: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_65, [32, 128, 1024]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_87: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_86, view_253);  mul_86 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_254: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_87, [4096, 1024]);  mul_87 = None
        permute_118: "f32[1024, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_66: "f32[4096, 512]" = torch.ops.aten.mm.default(view_254, permute_118);  view_254 = permute_118 = None
        view_255: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_66, [32, 128, 512]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_82: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_78, view_255);  add_78 = view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_30: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_82, 2)
        mean_20: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_83: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_88: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_82, rsqrt_20);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_89: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg92_1, mul_88);  arg92_1 = mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_256: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_89, [4096, 512])
        permute_119: "f32[512, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_67: "f32[4096, 384]" = torch.ops.aten.mm.default(view_256, permute_119);  view_256 = permute_119 = None
        view_257: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_67, [32, 128, 384]);  mm_67 = None
        view_258: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_257, [32, 128, -1, 64]);  view_257 = None

        # No stacktrace found for following nodes
        permute_default_45: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        mul_scalar_12: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_45, 1.0);  permute_default_45 = None
        expand_default_31: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_12, [32, 6, 128, 64]);  mul_scalar_12 = None
        clone_default_18: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_31, memory_format = torch.contiguous_format);  expand_default_31 = None
        view_default_36: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_18, [192, 128, 64]);  clone_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_259: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_89, [4096, 512])
        permute_121: "f32[512, 384]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_68: "f32[4096, 384]" = torch.ops.aten.mm.default(view_259, permute_121);  view_259 = permute_121 = None
        view_260: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_68, [32, 128, 384]);  mm_68 = None
        view_261: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_260, [32, 128, -1, 64]);  view_260 = None

        # No stacktrace found for following nodes
        permute_default_46: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        permute_default_48: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_46, [0, 1, 3, 2]);  permute_default_46 = None
        mul_scalar_13: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_48, 1.0);  permute_default_48 = None
        expand_default_32: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_13, [32, 6, 64, 128]);  mul_scalar_13 = None
        clone_default_19: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_32, memory_format = torch.contiguous_format);  expand_default_32 = None
        view_default_37: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_19, [192, 64, 128]);  clone_default_19 = None
        bmm_default_12: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_36, view_default_37);  view_default_36 = view_default_37 = None
        view_default_38: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_12, [32, 6, 128, 128]);  bmm_default_12 = None
        add_tensor_6: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_38, add_72);  view_default_38 = None
        eq_scalar_6: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_6, -inf)
        logical_not_default_12: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_6);  eq_scalar_6 = None
        any_dim_6: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_12, -1, True);  logical_not_default_12 = None
        logical_not_default_13: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_6);  any_dim_6 = None
        full_default_18: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_6: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_6, [-1], True)
        sub_tensor_6: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_6, amax_default_6);  add_tensor_6 = amax_default_6 = None
        exp_default_6: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_6);  sub_tensor_6 = None
        sum_dim_int_list_6: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_6, [-1], True)
        div_tensor_6: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_6, sum_dim_int_list_6);  exp_default_6 = sum_dim_int_list_6 = None
        where_self_6: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_13, full_default_18, div_tensor_6);  logical_not_default_13 = full_default_18 = div_tensor_6 = None
        expand_default_33: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_6, [32, 6, 128, 128]);  where_self_6 = None
        view_default_39: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_33, [192, 128, 128]);  expand_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_262: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_89, [4096, 512]);  mul_89 = None
        permute_123: "f32[512, 384]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_69: "f32[4096, 384]" = torch.ops.aten.mm.default(view_262, permute_123);  view_262 = permute_123 = None
        view_263: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_69, [32, 128, 384]);  mm_69 = None
        view_264: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_263, [32, 128, -1, 64]);  view_263 = None

        # No stacktrace found for following nodes
        permute_default_47: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        expand_default_34: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_47, [32, 6, 128, 64]);  permute_default_47 = None
        clone_default_20: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_34, memory_format = torch.contiguous_format);  expand_default_34 = None
        view_default_40: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_20, [192, 128, 64]);  clone_default_20 = None
        bmm_default_13: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_39, view_default_40);  view_default_39 = view_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_41: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_13, [32, 6, 128, 64]);  bmm_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_126: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_41, [0, 2, 1, 3]);  view_default_41 = None
        clone_85: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_273: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_85, [32, 128, -1]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_274: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_273, [4096, 384]);  view_273 = None
        permute_127: "f32[384, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_70: "f32[4096, 512]" = torch.ops.aten.mm.default(view_274, permute_127);  view_274 = permute_127 = None
        view_275: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_70, [32, 128, 512]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_85: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_82, view_275);  add_82 = view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_31: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_85, 2)
        mean_21: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_86: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_90: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, rsqrt_21);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_91: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg97_1, mul_90);  arg97_1 = mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_276: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_91, [4096, 512]);  mul_91 = None
        permute_128: "f32[512, 384]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_71: "f32[4096, 384]" = torch.ops.aten.mm.default(view_276, permute_128);  view_276 = permute_128 = None
        view_277: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_71, [32, 128, 384]);  mm_71 = None
        view_278: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_277, [32, 128, -1, 64]);  view_277 = None

        # No stacktrace found for following nodes
        permute_default_42: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_279: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_130: "f32[512, 384]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_72: "f32[4096, 384]" = torch.ops.aten.mm.default(view_279, permute_130);  view_279 = permute_130 = None
        view_280: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_72, [32, 128, 384]);  mm_72 = None
        view_281: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_280, [32, 128, -1, 64]);  view_280 = None

        # No stacktrace found for following nodes
        permute_default_43: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_282: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_132: "f32[512, 384]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_73: "f32[4096, 384]" = torch.ops.aten.mm.default(view_282, permute_132);  view_282 = permute_132 = None
        view_283: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_73, [32, 128, 384]);  mm_73 = None
        view_284: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_283, [32, 128, -1, 64]);  view_283 = None

        # No stacktrace found for following nodes
        permute_default_44: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        expand_default_30: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_42, permute_default_43, permute_default_44, expand_default_30, False, scale = 1.0);  permute_default_42 = permute_default_43 = permute_default_44 = expand_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_6: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
        clone_91: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_293: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_91, [32, 128, -1]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_294: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_293, [4096, 384]);  view_293 = None
        permute_136: "f32[384, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_74: "f32[4096, 512]" = torch.ops.aten.mm.default(view_294, permute_136);  view_294 = permute_136 = None
        view_295: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_74, [32, 128, 512]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_88: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_85, view_295);  add_85 = view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_32: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_88, 2)
        mean_22: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_89: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_92: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_88, rsqrt_22);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_93: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg102_1, mul_92);  arg102_1 = mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_296: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_93, [4096, 512])
        permute_137: "f32[512, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_75: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_296, permute_137);  view_296 = permute_137 = None
        view_297: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_75, [32, 128, 1024]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_94: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
        pow_33: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_297, 3.0)
        mul_95: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_90: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_297, mul_95);  view_297 = mul_95 = None
        mul_96: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_90, 0.7978845608028654);  add_90 = None
        tanh_9: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_96);  mul_96 = None
        add_91: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_97: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_94, add_91);  mul_94 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_298: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_93, [4096, 512]);  mul_93 = None
        permute_138: "f32[512, 1024]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_76: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_298, permute_138);  view_298 = permute_138 = None
        view_299: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_76, [32, 128, 1024]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_98: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, view_299);  mul_97 = view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_300: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_98, [4096, 1024]);  mul_98 = None
        permute_139: "f32[1024, 512]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_77: "f32[4096, 512]" = torch.ops.aten.mm.default(view_300, permute_139);  view_300 = permute_139 = None
        view_301: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_77, [32, 128, 512]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_92: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_88, view_301);  add_88 = view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_34: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 2)
        mean_23: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_99: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_23);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_100: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg106_1, mul_99);  arg106_1 = mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_302: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_100, [4096, 512])
        permute_140: "f32[512, 384]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_78: "f32[4096, 384]" = torch.ops.aten.mm.default(view_302, permute_140);  view_302 = permute_140 = None
        view_303: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_78, [32, 128, 384]);  mm_78 = None
        view_304: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_303, [32, 128, -1, 64]);  view_303 = None

        # No stacktrace found for following nodes
        permute_default_38: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        mul_scalar_10: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_38, 1.0);  permute_default_38 = None
        expand_default_26: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_10, [32, 6, 128, 64]);  mul_scalar_10 = None
        clone_default_15: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_26, memory_format = torch.contiguous_format);  expand_default_26 = None
        view_default_30: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_15, [192, 128, 64]);  clone_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_305: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_100, [4096, 512])
        permute_142: "f32[512, 384]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_79: "f32[4096, 384]" = torch.ops.aten.mm.default(view_305, permute_142);  view_305 = permute_142 = None
        view_306: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_79, [32, 128, 384]);  mm_79 = None
        view_307: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_306, [32, 128, -1, 64]);  view_306 = None

        # No stacktrace found for following nodes
        permute_default_39: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        permute_default_41: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_39, [0, 1, 3, 2]);  permute_default_39 = None
        mul_scalar_11: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_41, 1.0);  permute_default_41 = None
        expand_default_27: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_11, [32, 6, 64, 128]);  mul_scalar_11 = None
        clone_default_16: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_27, memory_format = torch.contiguous_format);  expand_default_27 = None
        view_default_31: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_16, [192, 64, 128]);  clone_default_16 = None
        bmm_default_10: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_30, view_default_31);  view_default_30 = view_default_31 = None
        view_default_32: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_10, [32, 6, 128, 128]);  bmm_default_10 = None
        add_tensor_5: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_32, add_72);  view_default_32 = None
        eq_scalar_5: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_5, -inf)
        logical_not_default_10: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_5);  eq_scalar_5 = None
        any_dim_5: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_10, -1, True);  logical_not_default_10 = None
        logical_not_default_11: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_5);  any_dim_5 = None
        full_default_17: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_5: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_5, [-1], True)
        sub_tensor_5: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_5, amax_default_5);  add_tensor_5 = amax_default_5 = None
        exp_default_5: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_5);  sub_tensor_5 = None
        sum_dim_int_list_5: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_5, [-1], True)
        div_tensor_5: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_5, sum_dim_int_list_5);  exp_default_5 = sum_dim_int_list_5 = None
        where_self_5: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_11, full_default_17, div_tensor_5);  logical_not_default_11 = full_default_17 = div_tensor_5 = None
        expand_default_28: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_5, [32, 6, 128, 128]);  where_self_5 = None
        view_default_33: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_28, [192, 128, 128]);  expand_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_308: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_100, [4096, 512]);  mul_100 = None
        permute_144: "f32[512, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_80: "f32[4096, 384]" = torch.ops.aten.mm.default(view_308, permute_144);  view_308 = permute_144 = None
        view_309: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_80, [32, 128, 384]);  mm_80 = None
        view_310: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_309, [32, 128, -1, 64]);  view_309 = None

        # No stacktrace found for following nodes
        permute_default_40: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        expand_default_29: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_40, [32, 6, 128, 64]);  permute_default_40 = None
        clone_default_17: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_29, memory_format = torch.contiguous_format);  expand_default_29 = None
        view_default_34: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_17, [192, 128, 64]);  clone_default_17 = None
        bmm_default_11: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_33, view_default_34);  view_default_33 = view_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_35: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_11, [32, 6, 128, 64]);  bmm_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_147: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_35, [0, 2, 1, 3]);  view_default_35 = None
        clone_99: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_319: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_99, [32, 128, -1]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_320: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_319, [4096, 384]);  view_319 = None
        permute_148: "f32[384, 512]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_81: "f32[4096, 512]" = torch.ops.aten.mm.default(view_320, permute_148);  view_320 = permute_148 = None
        view_321: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_81, [32, 128, 512]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_95: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_92, view_321);  add_92 = view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_35: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_95, 2)
        mean_24: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_96: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_101: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_95, rsqrt_24);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_102: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg111_1, mul_101);  arg111_1 = mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_322: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_102, [4096, 512]);  mul_102 = None
        permute_149: "f32[512, 384]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_82: "f32[4096, 384]" = torch.ops.aten.mm.default(view_322, permute_149);  view_322 = permute_149 = None
        view_323: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_82, [32, 128, 384]);  mm_82 = None
        view_324: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_323, [32, 128, -1, 64]);  view_323 = None

        # No stacktrace found for following nodes
        permute_default_35: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_325: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_151: "f32[512, 384]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_83: "f32[4096, 384]" = torch.ops.aten.mm.default(view_325, permute_151);  view_325 = permute_151 = None
        view_326: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_83, [32, 128, 384]);  mm_83 = None
        view_327: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_326, [32, 128, -1, 64]);  view_326 = None

        # No stacktrace found for following nodes
        permute_default_36: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_328: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_153: "f32[512, 384]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_84: "f32[4096, 384]" = torch.ops.aten.mm.default(view_328, permute_153);  view_328 = permute_153 = None
        view_329: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_84, [32, 128, 384]);  mm_84 = None
        view_330: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_329, [32, 128, -1, 64]);  view_329 = None

        # No stacktrace found for following nodes
        permute_default_37: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        expand_default_25: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_35, permute_default_36, permute_default_37, expand_default_25, False, scale = 1.0);  permute_default_35 = permute_default_36 = permute_default_37 = expand_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_5: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_156: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        clone_105: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_339: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_105, [32, 128, -1]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_340: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_339, [4096, 384]);  view_339 = None
        permute_157: "f32[384, 512]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_85: "f32[4096, 512]" = torch.ops.aten.mm.default(view_340, permute_157);  view_340 = permute_157 = None
        view_341: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_85, [32, 128, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_98: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_95, view_341);  add_95 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_36: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_98, 2)
        mean_25: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_99: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_103: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_98, rsqrt_25);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_104: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg116_1, mul_103);  arg116_1 = mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_342: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_104, [4096, 512])
        permute_158: "f32[512, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_86: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_342, permute_158);  view_342 = permute_158 = None
        view_343: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_86, [32, 128, 1024]);  mm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_105: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
        pow_37: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_343, 3.0)
        mul_106: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_100: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_343, mul_106);  view_343 = mul_106 = None
        mul_107: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_107);  mul_107 = None
        add_101: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_108: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_105, add_101);  mul_105 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_344: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_104, [4096, 512]);  mul_104 = None
        permute_159: "f32[512, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_87: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_344, permute_159);  view_344 = permute_159 = None
        view_345: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_87, [32, 128, 1024]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_109: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_108, view_345);  mul_108 = view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_346: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_109, [4096, 1024]);  mul_109 = None
        permute_160: "f32[1024, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_88: "f32[4096, 512]" = torch.ops.aten.mm.default(view_346, permute_160);  view_346 = permute_160 = None
        view_347: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_88, [32, 128, 512]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_102: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_98, view_347);  add_98 = view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_38: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_102, 2)
        mean_26: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_103: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_110: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_102, rsqrt_26);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_111: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg120_1, mul_110);  arg120_1 = mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_348: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_111, [4096, 512])
        permute_161: "f32[512, 384]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_89: "f32[4096, 384]" = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
        view_349: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_89, [32, 128, 384]);  mm_89 = None
        view_350: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_349, [32, 128, -1, 64]);  view_349 = None

        # No stacktrace found for following nodes
        permute_default_31: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        mul_scalar_8: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_31, 1.0);  permute_default_31 = None
        expand_default_21: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_8, [32, 6, 128, 64]);  mul_scalar_8 = None
        clone_default_12: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_21, memory_format = torch.contiguous_format);  expand_default_21 = None
        view_default_24: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_12, [192, 128, 64]);  clone_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_351: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_111, [4096, 512])
        permute_163: "f32[512, 384]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_90: "f32[4096, 384]" = torch.ops.aten.mm.default(view_351, permute_163);  view_351 = permute_163 = None
        view_352: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_90, [32, 128, 384]);  mm_90 = None
        view_353: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_352, [32, 128, -1, 64]);  view_352 = None

        # No stacktrace found for following nodes
        permute_default_32: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        permute_default_34: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_32, [0, 1, 3, 2]);  permute_default_32 = None
        mul_scalar_9: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_34, 1.0);  permute_default_34 = None
        expand_default_22: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_9, [32, 6, 64, 128]);  mul_scalar_9 = None
        clone_default_13: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_22, memory_format = torch.contiguous_format);  expand_default_22 = None
        view_default_25: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_13, [192, 64, 128]);  clone_default_13 = None
        bmm_default_8: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_24, view_default_25);  view_default_24 = view_default_25 = None
        view_default_26: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_8, [32, 6, 128, 128]);  bmm_default_8 = None
        add_tensor_4: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_26, add_72);  view_default_26 = None
        eq_scalar_4: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_4, -inf)
        logical_not_default_8: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_4);  eq_scalar_4 = None
        any_dim_4: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_8, -1, True);  logical_not_default_8 = None
        logical_not_default_9: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_4);  any_dim_4 = None
        full_default_16: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_4: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_4, [-1], True)
        sub_tensor_4: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_4, amax_default_4);  add_tensor_4 = amax_default_4 = None
        exp_default_4: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
        sum_dim_int_list_4: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_4, [-1], True)
        div_tensor_4: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_4, sum_dim_int_list_4);  exp_default_4 = sum_dim_int_list_4 = None
        where_self_4: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_9, full_default_16, div_tensor_4);  logical_not_default_9 = full_default_16 = div_tensor_4 = None
        expand_default_23: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_4, [32, 6, 128, 128]);  where_self_4 = None
        view_default_27: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_23, [192, 128, 128]);  expand_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_354: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_111, [4096, 512]);  mul_111 = None
        permute_165: "f32[512, 384]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_91: "f32[4096, 384]" = torch.ops.aten.mm.default(view_354, permute_165);  view_354 = permute_165 = None
        view_355: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_91, [32, 128, 384]);  mm_91 = None
        view_356: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_355, [32, 128, -1, 64]);  view_355 = None

        # No stacktrace found for following nodes
        permute_default_33: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        expand_default_24: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_33, [32, 6, 128, 64]);  permute_default_33 = None
        clone_default_14: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_24, memory_format = torch.contiguous_format);  expand_default_24 = None
        view_default_28: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_14, [192, 128, 64]);  clone_default_14 = None
        bmm_default_9: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_27, view_default_28);  view_default_27 = view_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_29: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_9, [32, 6, 128, 64]);  bmm_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_168: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_29, [0, 2, 1, 3]);  view_default_29 = None
        clone_113: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_365: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_113, [32, 128, -1]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_366: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_365, [4096, 384]);  view_365 = None
        permute_169: "f32[384, 512]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_92: "f32[4096, 512]" = torch.ops.aten.mm.default(view_366, permute_169);  view_366 = permute_169 = None
        view_367: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_92, [32, 128, 512]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_105: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_102, view_367);  add_102 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_39: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_105, 2)
        mean_27: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_106: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_112: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_105, rsqrt_27);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_113: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg125_1, mul_112);  arg125_1 = mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_368: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_113, [4096, 512]);  mul_113 = None
        permute_170: "f32[512, 384]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_93: "f32[4096, 384]" = torch.ops.aten.mm.default(view_368, permute_170);  view_368 = permute_170 = None
        view_369: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_93, [32, 128, 384]);  mm_93 = None
        view_370: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_369, [32, 128, -1, 64]);  view_369 = None

        # No stacktrace found for following nodes
        permute_default_28: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_371: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_172: "f32[512, 384]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_94: "f32[4096, 384]" = torch.ops.aten.mm.default(view_371, permute_172);  view_371 = permute_172 = None
        view_372: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_94, [32, 128, 384]);  mm_94 = None
        view_373: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_372, [32, 128, -1, 64]);  view_372 = None

        # No stacktrace found for following nodes
        permute_default_29: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_374: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_174: "f32[512, 384]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_95: "f32[4096, 384]" = torch.ops.aten.mm.default(view_374, permute_174);  view_374 = permute_174 = None
        view_375: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_95, [32, 128, 384]);  mm_95 = None
        view_376: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_375, [32, 128, -1, 64]);  view_375 = None

        # No stacktrace found for following nodes
        permute_default_30: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        expand_default_20: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_28, permute_default_29, permute_default_30, expand_default_20, False, scale = 1.0);  permute_default_28 = permute_default_29 = permute_default_30 = expand_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_4: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_177: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        clone_119: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_385: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_119, [32, 128, -1]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_386: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_385, [4096, 384]);  view_385 = None
        permute_178: "f32[384, 512]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_96: "f32[4096, 512]" = torch.ops.aten.mm.default(view_386, permute_178);  view_386 = permute_178 = None
        view_387: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_96, [32, 128, 512]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_108: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_105, view_387);  add_105 = view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_40: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_108, 2)
        mean_28: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_109: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_114: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_108, rsqrt_28);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_115: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg130_1, mul_114);  arg130_1 = mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_388: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_115, [4096, 512])
        permute_179: "f32[512, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_97: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_388, permute_179);  view_388 = permute_179 = None
        view_389: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_97, [32, 128, 1024]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        pow_41: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
        mul_117: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_110: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_389, mul_117);  view_389 = mul_117 = None
        mul_118: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
        tanh_11: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_111: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_116, add_111);  mul_116 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_390: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_115, [4096, 512]);  mul_115 = None
        permute_180: "f32[512, 1024]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_98: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_390, permute_180);  view_390 = permute_180 = None
        view_391: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_98, [32, 128, 1024]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_120: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_119, view_391);  mul_119 = view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_392: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_120, [4096, 1024]);  mul_120 = None
        permute_181: "f32[1024, 512]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_99: "f32[4096, 512]" = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
        view_393: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_99, [32, 128, 512]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_112: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_108, view_393);  add_108 = view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_42: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_112, 2)
        mean_29: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_113: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_121: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_112, rsqrt_29);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_122: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg134_1, mul_121);  arg134_1 = mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_394: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_122, [4096, 512])
        permute_182: "f32[512, 384]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_100: "f32[4096, 384]" = torch.ops.aten.mm.default(view_394, permute_182);  view_394 = permute_182 = None
        view_395: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_100, [32, 128, 384]);  mm_100 = None
        view_396: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_395, [32, 128, -1, 64]);  view_395 = None

        # No stacktrace found for following nodes
        permute_default_24: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
        mul_scalar_6: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_24, 1.0);  permute_default_24 = None
        expand_default_16: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_6, [32, 6, 128, 64]);  mul_scalar_6 = None
        clone_default_9: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_16, memory_format = torch.contiguous_format);  expand_default_16 = None
        view_default_18: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_9, [192, 128, 64]);  clone_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_397: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_122, [4096, 512])
        permute_184: "f32[512, 384]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_101: "f32[4096, 384]" = torch.ops.aten.mm.default(view_397, permute_184);  view_397 = permute_184 = None
        view_398: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_101, [32, 128, 384]);  mm_101 = None
        view_399: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_398, [32, 128, -1, 64]);  view_398 = None

        # No stacktrace found for following nodes
        permute_default_25: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        permute_default_27: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_25, [0, 1, 3, 2]);  permute_default_25 = None
        mul_scalar_7: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_27, 1.0);  permute_default_27 = None
        expand_default_17: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_7, [32, 6, 64, 128]);  mul_scalar_7 = None
        clone_default_10: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_17, memory_format = torch.contiguous_format);  expand_default_17 = None
        view_default_19: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_10, [192, 64, 128]);  clone_default_10 = None
        bmm_default_6: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_18, view_default_19);  view_default_18 = view_default_19 = None
        view_default_20: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_6, [32, 6, 128, 128]);  bmm_default_6 = None
        add_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_20, add_72);  view_default_20 = None
        eq_scalar_3: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_3, -inf)
        logical_not_default_6: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_3);  eq_scalar_3 = None
        any_dim_3: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_6, -1, True);  logical_not_default_6 = None
        logical_not_default_7: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_3);  any_dim_3 = None
        full_default_15: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_3: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_3, [-1], True)
        sub_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_3, amax_default_3);  add_tensor_3 = amax_default_3 = None
        exp_default_3: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_3);  sub_tensor_3 = None
        sum_dim_int_list_3: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_3, [-1], True)
        div_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_3, sum_dim_int_list_3);  exp_default_3 = sum_dim_int_list_3 = None
        where_self_3: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_7, full_default_15, div_tensor_3);  logical_not_default_7 = full_default_15 = div_tensor_3 = None
        expand_default_18: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_3, [32, 6, 128, 128]);  where_self_3 = None
        view_default_21: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_18, [192, 128, 128]);  expand_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_400: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_122, [4096, 512]);  mul_122 = None
        permute_186: "f32[512, 384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_102: "f32[4096, 384]" = torch.ops.aten.mm.default(view_400, permute_186);  view_400 = permute_186 = None
        view_401: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_102, [32, 128, 384]);  mm_102 = None
        view_402: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_401, [32, 128, -1, 64]);  view_401 = None

        # No stacktrace found for following nodes
        permute_default_26: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        expand_default_19: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_26, [32, 6, 128, 64]);  permute_default_26 = None
        clone_default_11: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_19, memory_format = torch.contiguous_format);  expand_default_19 = None
        view_default_22: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_11, [192, 128, 64]);  clone_default_11 = None
        bmm_default_7: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_21, view_default_22);  view_default_21 = view_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_23: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_7, [32, 6, 128, 64]);  bmm_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_189: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_23, [0, 2, 1, 3]);  view_default_23 = None
        clone_127: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_411: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_127, [32, 128, -1]);  clone_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_412: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_411, [4096, 384]);  view_411 = None
        permute_190: "f32[384, 512]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_103: "f32[4096, 512]" = torch.ops.aten.mm.default(view_412, permute_190);  view_412 = permute_190 = None
        view_413: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_103, [32, 128, 512]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_115: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_112, view_413);  add_112 = view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_43: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_115, 2)
        mean_30: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_116: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_123: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_115, rsqrt_30);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_124: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg139_1, mul_123);  arg139_1 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_414: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_124, [4096, 512]);  mul_124 = None
        permute_191: "f32[512, 384]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_104: "f32[4096, 384]" = torch.ops.aten.mm.default(view_414, permute_191);  view_414 = permute_191 = None
        view_415: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_104, [32, 128, 384]);  mm_104 = None
        view_416: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_415, [32, 128, -1, 64]);  view_415 = None

        # No stacktrace found for following nodes
        permute_default_21: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_417: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_193: "f32[512, 384]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_105: "f32[4096, 384]" = torch.ops.aten.mm.default(view_417, permute_193);  view_417 = permute_193 = None
        view_418: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_105, [32, 128, 384]);  mm_105 = None
        view_419: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_418, [32, 128, -1, 64]);  view_418 = None

        # No stacktrace found for following nodes
        permute_default_22: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_420: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_195: "f32[512, 384]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_106: "f32[4096, 384]" = torch.ops.aten.mm.default(view_420, permute_195);  view_420 = permute_195 = None
        view_421: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_106, [32, 128, 384]);  mm_106 = None
        view_422: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_421, [32, 128, -1, 64]);  view_421 = None

        # No stacktrace found for following nodes
        permute_default_23: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        expand_default_15: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, expand_default_15, False, scale = 1.0);  permute_default_21 = permute_default_22 = permute_default_23 = expand_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_3: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_198: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_3, [0, 2, 1, 3]);  getitem_3 = None
        clone_133: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_431: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_133, [32, 128, -1]);  clone_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_432: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_431, [4096, 384]);  view_431 = None
        permute_199: "f32[384, 512]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_107: "f32[4096, 512]" = torch.ops.aten.mm.default(view_432, permute_199);  view_432 = permute_199 = None
        view_433: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_107, [32, 128, 512]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_118: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_115, view_433);  add_115 = view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_44: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_118, 2)
        mean_31: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_119: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_125: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_118, rsqrt_31);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_126: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg144_1, mul_125);  arg144_1 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_434: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_126, [4096, 512])
        permute_200: "f32[512, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_108: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_434, permute_200);  view_434 = permute_200 = None
        view_435: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_108, [32, 128, 1024]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_127: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_435, 0.5)
        pow_45: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_435, 3.0)
        mul_128: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_120: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_435, mul_128);  view_435 = mul_128 = None
        mul_129: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_120, 0.7978845608028654);  add_120 = None
        tanh_12: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_129);  mul_129 = None
        add_121: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_130: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_127, add_121);  mul_127 = add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_436: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_126, [4096, 512]);  mul_126 = None
        permute_201: "f32[512, 1024]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_109: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_436, permute_201);  view_436 = permute_201 = None
        view_437: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_109, [32, 128, 1024]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_131: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_130, view_437);  mul_130 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_438: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_131, [4096, 1024]);  mul_131 = None
        permute_202: "f32[1024, 512]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_110: "f32[4096, 512]" = torch.ops.aten.mm.default(view_438, permute_202);  view_438 = permute_202 = None
        view_439: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_110, [32, 128, 512]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_122: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_118, view_439);  add_118 = view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_46: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_122, 2)
        mean_32: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_123: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_132: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_122, rsqrt_32);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_133: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg148_1, mul_132);  arg148_1 = mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_440: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_133, [4096, 512])
        permute_203: "f32[512, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_111: "f32[4096, 384]" = torch.ops.aten.mm.default(view_440, permute_203);  view_440 = permute_203 = None
        view_441: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_111, [32, 128, 384]);  mm_111 = None
        view_442: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_441, [32, 128, -1, 64]);  view_441 = None

        # No stacktrace found for following nodes
        permute_default_17: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        mul_scalar_4: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_17, 1.0);  permute_default_17 = None
        expand_default_11: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_4, [32, 6, 128, 64]);  mul_scalar_4 = None
        clone_default_6: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_11, memory_format = torch.contiguous_format);  expand_default_11 = None
        view_default_12: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_6, [192, 128, 64]);  clone_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_443: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_133, [4096, 512])
        permute_205: "f32[512, 384]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_112: "f32[4096, 384]" = torch.ops.aten.mm.default(view_443, permute_205);  view_443 = permute_205 = None
        view_444: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_112, [32, 128, 384]);  mm_112 = None
        view_445: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_444, [32, 128, -1, 64]);  view_444 = None

        # No stacktrace found for following nodes
        permute_default_18: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        permute_default_20: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_18, [0, 1, 3, 2]);  permute_default_18 = None
        mul_scalar_5: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_20, 1.0);  permute_default_20 = None
        expand_default_12: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_5, [32, 6, 64, 128]);  mul_scalar_5 = None
        clone_default_7: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_12, memory_format = torch.contiguous_format);  expand_default_12 = None
        view_default_13: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_7, [192, 64, 128]);  clone_default_7 = None
        bmm_default_4: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_12, view_default_13);  view_default_12 = view_default_13 = None
        view_default_14: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_4, [32, 6, 128, 128]);  bmm_default_4 = None
        add_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_14, add_72);  view_default_14 = None
        eq_scalar_2: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_2, -inf)
        logical_not_default_4: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_2);  eq_scalar_2 = None
        any_dim_2: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_4, -1, True);  logical_not_default_4 = None
        logical_not_default_5: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_2);  any_dim_2 = None
        full_default_14: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_2: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_2, [-1], True)
        sub_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_2, amax_default_2);  add_tensor_2 = amax_default_2 = None
        exp_default_2: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        sum_dim_int_list_2: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_2, [-1], True)
        div_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_2, sum_dim_int_list_2);  exp_default_2 = sum_dim_int_list_2 = None
        where_self_2: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_5, full_default_14, div_tensor_2);  logical_not_default_5 = full_default_14 = div_tensor_2 = None
        expand_default_13: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_2, [32, 6, 128, 128]);  where_self_2 = None
        view_default_15: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_13, [192, 128, 128]);  expand_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_446: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_133, [4096, 512]);  mul_133 = None
        permute_207: "f32[512, 384]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_113: "f32[4096, 384]" = torch.ops.aten.mm.default(view_446, permute_207);  view_446 = permute_207 = None
        view_447: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_113, [32, 128, 384]);  mm_113 = None
        view_448: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_447, [32, 128, -1, 64]);  view_447 = None

        # No stacktrace found for following nodes
        permute_default_19: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        expand_default_14: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_19, [32, 6, 128, 64]);  permute_default_19 = None
        clone_default_8: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_14, memory_format = torch.contiguous_format);  expand_default_14 = None
        view_default_16: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_8, [192, 128, 64]);  clone_default_8 = None
        bmm_default_5: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_15, view_default_16);  view_default_15 = view_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_17: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_5, [32, 6, 128, 64]);  bmm_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_17, [0, 2, 1, 3]);  view_default_17 = None
        clone_141: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_457: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_141, [32, 128, -1]);  clone_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_458: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_457, [4096, 384]);  view_457 = None
        permute_211: "f32[384, 512]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_114: "f32[4096, 512]" = torch.ops.aten.mm.default(view_458, permute_211);  view_458 = permute_211 = None
        view_459: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_114, [32, 128, 512]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_125: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_122, view_459);  add_122 = view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_47: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_125, 2)
        mean_33: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_126: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_134: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_125, rsqrt_33);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_135: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg153_1, mul_134);  arg153_1 = mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_460: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_135, [4096, 512]);  mul_135 = None
        permute_212: "f32[512, 384]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_115: "f32[4096, 384]" = torch.ops.aten.mm.default(view_460, permute_212);  view_460 = permute_212 = None
        view_461: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_115, [32, 128, 384]);  mm_115 = None
        view_462: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_461, [32, 128, -1, 64]);  view_461 = None

        # No stacktrace found for following nodes
        permute_default_14: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_463: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_214: "f32[512, 384]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_116: "f32[4096, 384]" = torch.ops.aten.mm.default(view_463, permute_214);  view_463 = permute_214 = None
        view_464: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_116, [32, 128, 384]);  mm_116 = None
        view_465: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_464, [32, 128, -1, 64]);  view_464 = None

        # No stacktrace found for following nodes
        permute_default_15: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_466: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_216: "f32[512, 384]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_117: "f32[4096, 384]" = torch.ops.aten.mm.default(view_466, permute_216);  view_466 = permute_216 = None
        view_467: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_117, [32, 128, 384]);  mm_117 = None
        view_468: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_467, [32, 128, -1, 64]);  view_467 = None

        # No stacktrace found for following nodes
        permute_default_16: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        expand_default_10: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_14, permute_default_15, permute_default_16, expand_default_10, False, scale = 1.0);  permute_default_14 = permute_default_15 = permute_default_16 = expand_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_2: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_219: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_147: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_477: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_147, [32, 128, -1]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_478: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_477, [4096, 384]);  view_477 = None
        permute_220: "f32[384, 512]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        mm_118: "f32[4096, 512]" = torch.ops.aten.mm.default(view_478, permute_220);  view_478 = permute_220 = None
        view_479: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_118, [32, 128, 512]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_128: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_125, view_479);  add_125 = view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_48: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_128, 2)
        mean_34: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_129: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_136: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_128, rsqrt_34);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_137: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg158_1, mul_136);  arg158_1 = mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_480: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_137, [4096, 512])
        permute_221: "f32[512, 1024]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_119: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_480, permute_221);  view_480 = permute_221 = None
        view_481: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_119, [32, 128, 1024]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_138: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        pow_49: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_139: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_130: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_481, mul_139);  view_481 = mul_139 = None
        mul_140: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_130, 0.7978845608028654);  add_130 = None
        tanh_13: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_140);  mul_140 = None
        add_131: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_141: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_138, add_131);  mul_138 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_482: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_137, [4096, 512]);  mul_137 = None
        permute_222: "f32[512, 1024]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_120: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_482, permute_222);  view_482 = permute_222 = None
        view_483: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_120, [32, 128, 1024]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_142: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_141, view_483);  mul_141 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_484: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_142, [4096, 1024]);  mul_142 = None
        permute_223: "f32[1024, 512]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_121: "f32[4096, 512]" = torch.ops.aten.mm.default(view_484, permute_223);  view_484 = permute_223 = None
        view_485: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_121, [32, 128, 512]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_132: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_128, view_485);  add_128 = view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_50: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_132, 2)
        mean_35: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_143: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_132, rsqrt_35);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_144: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg162_1, mul_143);  arg162_1 = mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_486: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_144, [4096, 512])
        permute_224: "f32[512, 384]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_122: "f32[4096, 384]" = torch.ops.aten.mm.default(view_486, permute_224);  view_486 = permute_224 = None
        view_487: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_122, [32, 128, 384]);  mm_122 = None
        view_488: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_487, [32, 128, -1, 64]);  view_487 = None

        # No stacktrace found for following nodes
        permute_default_10: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        mul_scalar_2: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_10, 1.0);  permute_default_10 = None
        expand_default_6: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar_2, [32, 6, 128, 64]);  mul_scalar_2 = None
        clone_default_3: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_6, memory_format = torch.contiguous_format);  expand_default_6 = None
        view_default_6: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_3, [192, 128, 64]);  clone_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_489: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_144, [4096, 512])
        permute_226: "f32[512, 384]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_123: "f32[4096, 384]" = torch.ops.aten.mm.default(view_489, permute_226);  view_489 = permute_226 = None
        view_490: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_123, [32, 128, 384]);  mm_123 = None
        view_491: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_490, [32, 128, -1, 64]);  view_490 = None

        # No stacktrace found for following nodes
        permute_default_11: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        permute_default_13: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_11, [0, 1, 3, 2]);  permute_default_11 = None
        mul_scalar_3: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_13, 1.0);  permute_default_13 = None
        expand_default_7: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_3, [32, 6, 64, 128]);  mul_scalar_3 = None
        clone_default_4: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_7, memory_format = torch.contiguous_format);  expand_default_7 = None
        view_default_7: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_4, [192, 64, 128]);  clone_default_4 = None
        bmm_default_2: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default_6, view_default_7);  view_default_6 = view_default_7 = None
        view_default_8: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_2, [32, 6, 128, 128]);  bmm_default_2 = None
        add_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_8, add_72);  view_default_8 = None
        eq_scalar_1: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_1, -inf)
        logical_not_default_2: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar_1);  eq_scalar_1 = None
        any_dim_1: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default_2, -1, True);  logical_not_default_2 = None
        logical_not_default_3: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim_1);  any_dim_1 = None
        full_default_13: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default_1: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
        sub_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default_1);  add_tensor_1 = amax_default_1 = None
        exp_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True)
        div_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        where_self_1: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_3, full_default_13, div_tensor_1);  logical_not_default_3 = full_default_13 = div_tensor_1 = None
        expand_default_8: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_1, [32, 6, 128, 128]);  where_self_1 = None
        view_default_9: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_8, [192, 128, 128]);  expand_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_492: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_144, [4096, 512]);  mul_144 = None
        permute_228: "f32[512, 384]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_124: "f32[4096, 384]" = torch.ops.aten.mm.default(view_492, permute_228);  view_492 = permute_228 = None
        view_493: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_124, [32, 128, 384]);  mm_124 = None
        view_494: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_493, [32, 128, -1, 64]);  view_493 = None

        # No stacktrace found for following nodes
        permute_default_12: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        expand_default_9: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_12, [32, 6, 128, 64]);  permute_default_12 = None
        clone_default_5: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_9, memory_format = torch.contiguous_format);  expand_default_9 = None
        view_default_10: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_5, [192, 128, 64]);  clone_default_5 = None
        bmm_default_3: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_9, view_default_10);  view_default_9 = view_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_11: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_3, [32, 6, 128, 64]);  bmm_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_231: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_11, [0, 2, 1, 3]);  view_default_11 = None
        clone_155: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_503: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_155, [32, 128, -1]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_504: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_503, [4096, 384]);  view_503 = None
        permute_232: "f32[384, 512]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_125: "f32[4096, 512]" = torch.ops.aten.mm.default(view_504, permute_232);  view_504 = permute_232 = None
        view_505: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_125, [32, 128, 512]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_135: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_132, view_505);  add_132 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_51: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_135, 2)
        mean_36: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_145: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_135, rsqrt_36);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_146: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg167_1, mul_145);  arg167_1 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_506: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_146, [4096, 512]);  mul_146 = None
        permute_233: "f32[512, 384]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        mm_126: "f32[4096, 384]" = torch.ops.aten.mm.default(view_506, permute_233);  view_506 = permute_233 = None
        view_507: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_126, [32, 128, 384]);  mm_126 = None
        view_508: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_507, [32, 128, -1, 64]);  view_507 = None

        # No stacktrace found for following nodes
        permute_default_7: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_509: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_235: "f32[512, 384]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_127: "f32[4096, 384]" = torch.ops.aten.mm.default(view_509, permute_235);  view_509 = permute_235 = None
        view_510: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_127, [32, 128, 384]);  mm_127 = None
        view_511: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_510, [32, 128, -1, 64]);  view_510 = None

        # No stacktrace found for following nodes
        permute_default_8: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_512: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_237: "f32[512, 384]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_128: "f32[4096, 384]" = torch.ops.aten.mm.default(view_512, permute_237);  view_512 = permute_237 = None
        view_513: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_128, [32, 128, 384]);  mm_128 = None
        view_514: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_513, [32, 128, -1, 64]);  view_513 = None

        # No stacktrace found for following nodes
        permute_default_9: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        expand_default_5: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128])
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_7, permute_default_8, permute_default_9, expand_default_5, False, scale = 1.0);  permute_default_7 = permute_default_8 = permute_default_9 = expand_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem_1: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_240: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem_1, [0, 2, 1, 3]);  getitem_1 = None
        clone_161: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_523: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_161, [32, 128, -1]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_524: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_523, [4096, 384]);  view_523 = None
        permute_241: "f32[384, 512]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_129: "f32[4096, 512]" = torch.ops.aten.mm.default(view_524, permute_241);  view_524 = permute_241 = None
        view_525: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_129, [32, 128, 512]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_138: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_135, view_525);  add_135 = view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_52: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_138, 2)
        mean_37: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_147: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_138, rsqrt_37);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_148: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg172_1, mul_147);  arg172_1 = mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_526: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_148, [4096, 512])
        permute_242: "f32[512, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_130: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_526, permute_242);  view_526 = permute_242 = None
        view_527: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_130, [32, 128, 1024]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_149: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
        pow_53: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
        mul_150: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_140: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_527, mul_150);  view_527 = mul_150 = None
        mul_151: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
        tanh_14: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_151);  mul_151 = None
        add_141: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_152: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_149, add_141);  mul_149 = add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_528: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_148, [4096, 512]);  mul_148 = None
        permute_243: "f32[512, 1024]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_131: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_528, permute_243);  view_528 = permute_243 = None
        view_529: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_131, [32, 128, 1024]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_153: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_152, view_529);  mul_152 = view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_530: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_153, [4096, 1024]);  mul_153 = None
        permute_244: "f32[1024, 512]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_132: "f32[4096, 512]" = torch.ops.aten.mm.default(view_530, permute_244);  view_530 = permute_244 = None
        view_531: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_132, [32, 128, 512]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_142: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_138, view_531);  add_138 = view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_54: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_142, 2)
        mean_38: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_143: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_154: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_142, rsqrt_38);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_155: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg176_1, mul_154);  arg176_1 = mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_532: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_155, [4096, 512])
        permute_245: "f32[512, 384]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_133: "f32[4096, 384]" = torch.ops.aten.mm.default(view_532, permute_245);  view_532 = permute_245 = None
        view_533: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_133, [32, 128, 384]);  mm_133 = None
        view_534: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_533, [32, 128, -1, 64]);  view_533 = None

        # No stacktrace found for following nodes
        permute_default_3: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        mul_scalar: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default_3, 1.0);  permute_default_3 = None
        expand_default_1: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar, [32, 6, 128, 64]);  mul_scalar = None
        clone_default: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default, [192, 128, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_535: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_155, [4096, 512])
        permute_247: "f32[512, 384]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_134: "f32[4096, 384]" = torch.ops.aten.mm.default(view_535, permute_247);  view_535 = permute_247 = None
        view_536: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_134, [32, 128, 384]);  mm_134 = None
        view_537: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_536, [32, 128, -1, 64]);  view_536 = None

        # No stacktrace found for following nodes
        permute_default_4: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
        permute_default_6: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_4, [0, 1, 3, 2]);  permute_default_4 = None
        mul_scalar_1: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_6, 1.0);  permute_default_6 = None
        expand_default_2: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_1, [32, 6, 64, 128]);  mul_scalar_1 = None
        clone_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_1: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_1, [192, 64, 128]);  clone_default_1 = None
        bmm_default: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default, [32, 6, 128, 128]);  bmm_default = None
        add_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default_2, add_72);  view_default_2 = add_72 = None
        eq_scalar: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_12: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_12, div_tensor);  logical_not_default_1 = full_default_12 = div_tensor = None
        expand_default_3: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self, [32, 6, 128, 128]);  where_self = None
        view_default_3: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_3, [192, 128, 128]);  expand_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_538: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_155, [4096, 512]);  mul_155 = None
        permute_249: "f32[512, 384]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_135: "f32[4096, 384]" = torch.ops.aten.mm.default(view_538, permute_249);  view_538 = permute_249 = None
        view_539: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_135, [32, 128, 384]);  mm_135 = None
        view_540: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_539, [32, 128, -1, 64]);  view_539 = None

        # No stacktrace found for following nodes
        permute_default_5: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        expand_default_4: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_5, [32, 6, 128, 64]);  permute_default_5 = None
        clone_default_2: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_4, memory_format = torch.contiguous_format);  expand_default_4 = None
        view_default_4: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default_2, [192, 128, 64]);  clone_default_2 = None
        bmm_default_1: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        view_default_5: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_default_1, [32, 6, 128, 64]);  bmm_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_252: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_169: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_549: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_169, [32, 128, -1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_550: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_549, [4096, 384]);  view_549 = None
        permute_253: "f32[384, 512]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_136: "f32[4096, 512]" = torch.ops.aten.mm.default(view_550, permute_253);  view_550 = permute_253 = None
        view_551: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_136, [32, 128, 512]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_145: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_142, view_551);  add_142 = view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_55: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_145, 2)
        mean_39: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_146: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_156: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_145, rsqrt_39);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_157: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg181_1, mul_156);  arg181_1 = mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_552: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_157, [4096, 512]);  mul_157 = None
        permute_254: "f32[512, 384]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_137: "f32[4096, 384]" = torch.ops.aten.mm.default(view_552, permute_254);  view_552 = permute_254 = None
        view_553: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_137, [32, 128, 384]);  mm_137 = None
        view_554: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_553, [32, 128, -1, 64]);  view_553 = None

        # No stacktrace found for following nodes
        permute_default: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_555: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_256: "f32[512, 384]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_138: "f32[4096, 384]" = torch.ops.aten.mm.default(view_555, permute_256);  view_555 = permute_256 = None
        view_556: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_138, [32, 128, 384]);  mm_138 = None
        view_557: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_556, [32, 128, -1, 64]);  view_556 = None

        # No stacktrace found for following nodes
        permute_default_1: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_558: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_258: "f32[512, 384]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_139: "f32[4096, 384]" = torch.ops.aten.mm.default(view_558, permute_258);  view_558 = permute_258 = None
        view_559: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_139, [32, 128, 384]);  mm_139 = None
        view_560: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_559, [32, 128, -1, 64]);  view_559 = None

        # No stacktrace found for following nodes
        permute_default_2: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        expand_default: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_76, [32, 6, 128, 128]);  add_76 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, expand_default, False, scale = 1.0);  permute_default = permute_default_1 = permute_default_2 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        getitem: "f32[32, 6, 128, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_581: "i64[4096]" = torch.ops.aten.reshape.default(arg76_1, [-1]);  arg76_1 = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_581, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_261: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        clone_175: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_569: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_175, [32, 128, -1]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_570: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_569, [4096, 384]);  view_569 = None
        permute_262: "f32[384, 512]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_140: "f32[4096, 512]" = torch.ops.aten.mm.default(view_570, permute_262);  view_570 = permute_262 = None
        view_571: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_140, [32, 128, 512]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_148: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_145, view_571);  add_145 = view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_56: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_148, 2)
        mean_40: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_149: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_158: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_148, rsqrt_40);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_159: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg186_1, mul_158);  arg186_1 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_572: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_159, [4096, 512])
        permute_263: "f32[512, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_141: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_572, permute_263);  view_572 = permute_263 = None
        view_573: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_141, [32, 128, 1024]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_160: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_573, 0.5)
        pow_57: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_573, 3.0)
        mul_161: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_150: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_573, mul_161);  view_573 = mul_161 = None
        mul_162: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
        tanh_15: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_162);  mul_162 = None
        add_151: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_163: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_160, add_151);  mul_160 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_574: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_159, [4096, 512]);  mul_159 = None
        permute_264: "f32[512, 1024]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_142: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_574, permute_264);  view_574 = permute_264 = None
        view_575: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_142, [32, 128, 1024]);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_164: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_163, view_575);  mul_163 = view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_576: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_164, [4096, 1024]);  mul_164 = None
        permute_265: "f32[1024, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_143: "f32[4096, 512]" = torch.ops.aten.mm.default(view_576, permute_265);  view_576 = permute_265 = None
        view_577: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_143, [32, 128, 512]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_152: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_148, view_577);  add_148 = view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_58: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_152, 2)
        mean_41: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_153: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_165: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_152, rsqrt_41);  add_152 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_166: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg190_1, mul_165);  arg190_1 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_578: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_166, [4096, 512]);  mul_166 = None
        permute_266: "f32[512, 250112]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_144: "f32[4096, 250112]" = torch.ops.aten.mm.default(view_578, permute_266);  view_578 = permute_266 = None
        view_579: "f32[32, 128, 250112]" = torch.ops.aten.reshape.default(mm_144, [32, 128, 250112]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_580: "f32[4096, 250112]" = torch.ops.aten.reshape.default(view_579, [-1, 250112])
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(view_580, [1], True)
        sub_26: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(view_580, amax_24);  view_580 = amax_24 = None
        exp_24: "f32[4096, 250112]" = torch.ops.aten.exp.default(sub_26)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log_2: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_27: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub_26, log_2);  sub_26 = log_2 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_581, -100)
        full_default_10: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "i64[4096]" = torch.ops.aten.where.self(ne, view_581, full_default_10);  ne = full_default_10 = None
        unsqueeze_18: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_5, 1);  where_5 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_18);  sub_27 = unsqueeze_18 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[4096]" = torch.ops.aten.where.self(ne_1, neg_1, full_default_11);  ne_1 = neg_1 = full_default_11 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_6);  where_6 = None
        ne_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view_581, -100);  view_581 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_5: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        div_28: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_5);  sum_27 = convert_element_type_5 = None
        return (div_28, view_579, mul_75)
