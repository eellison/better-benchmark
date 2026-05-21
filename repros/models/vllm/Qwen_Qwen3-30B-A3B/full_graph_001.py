class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 512]", primals_3: "f32[64]", primals_4: "bf16[2048]", primals_5: "bf16[4096, 2048]", primals_6: "bf16[128]", primals_7: "bf16[512, 2048]", primals_8: "bf16[128]", primals_9: "bf16[512, 2048]", primals_10: "bf16[2048, 4096]", primals_11: "bf16[2048]", primals_12: "bf16[128, 2048]", primals_13: "bf16[128, 1536, 2048]", primals_14: "bf16[128, 2048, 768]", primals_15: "bf16[2048]", primals_16: "bf16[4096, 2048]", primals_17: "bf16[128]", primals_18: "bf16[512, 2048]", primals_19: "bf16[128]", primals_20: "bf16[512, 2048]", primals_21: "bf16[2048, 4096]", primals_22: "bf16[2048]", primals_23: "bf16[128, 2048]", primals_24: "bf16[128, 1536, 2048]", primals_25: "bf16[128, 2048, 768]", primals_26: "bf16[2048]", primals_27: "bf16[4096, 2048]", primals_28: "bf16[128]", primals_29: "bf16[512, 2048]", primals_30: "bf16[128]", primals_31: "bf16[512, 2048]", primals_32: "bf16[2048, 4096]", primals_33: "bf16[2048]", primals_34: "bf16[128, 2048]", primals_35: "bf16[128, 1536, 2048]", primals_36: "bf16[128, 2048, 768]", primals_37: "bf16[2048]", primals_38: "bf16[4096, 2048]", primals_39: "bf16[128]", primals_40: "bf16[512, 2048]", primals_41: "bf16[128]", primals_42: "bf16[512, 2048]", primals_43: "bf16[2048, 4096]", primals_44: "bf16[2048]", primals_45: "bf16[128, 2048]", primals_46: "bf16[128, 1536, 2048]", primals_47: "bf16[128, 2048, 768]", primals_48: "bf16[2048]", primals_49: "bf16[151936, 2048]", embedding: "bf16[4, 512, 2048]", rsqrt: "f32[4, 512, 1]", view_4: "bf16[2048, 2048]", mm: "bf16[2048, 4096]", rsqrt_1: "f32[4, 512, 32, 1]", mm_1: "bf16[2048, 512]", rsqrt_2: "f32[4, 512, 4, 1]", add_6: "bf16[4, 32, 512, 128]", view_13: "bf16[4, 32, 512, 128]", view_14: "bf16[4, 32, 512, 128]", where: "bf16[4, 1, 512, 512]", getitem: "bf16[4, 32, 512, 128]", getitem_1: "f32[4, 32, 512, 1]", getitem_6: "i64[]", getitem_7: "i64[]", mm_3: "bf16[2048, 2048]", rsqrt_3: "f32[4, 512, 1]", view_18: "bf16[2048, 2048]", mm_4: "bf16[2048, 128]", amax: "f32[2048, 1]", sum_1: "f32[2048, 1]", getitem_10: "i64[2048, 8]", sum_2: "f32[2048, 1]", div_1: "f32[2048, 8]", getitem_12: "i64[16384]", div_2: "i64[16384]", cumsum_1: "i32[128]", unsqueeze_18: "b8[16384, 1]", where_1: "bf16[16384, 2048]", _grouped_mm: "bf16[16384, 1536]", mul_15: "bf16[16384, 768]", _grouped_mm_1: "bf16[16384, 2048]", index_put: "i64[16384]", add_11: "bf16[4, 512, 2048]", rsqrt_4: "f32[4, 512, 1]", view_24: "bf16[2048, 2048]", mm_5: "bf16[2048, 4096]", rsqrt_5: "f32[4, 512, 32, 1]", mm_6: "bf16[2048, 512]", rsqrt_6: "f32[4, 512, 4, 1]", add_15: "bf16[4, 32, 512, 128]", view_33: "bf16[4, 32, 512, 128]", view_34: "bf16[4, 32, 512, 128]", getitem_15: "bf16[4, 32, 512, 128]", getitem_16: "f32[4, 32, 512, 1]", getitem_21: "i64[]", getitem_22: "i64[]", add_17: "bf16[4, 512, 2048]", rsqrt_7: "f32[4, 512, 1]", view_38: "bf16[2048, 2048]", mm_9: "bf16[2048, 128]", amax_1: "f32[2048, 1]", sum_4: "f32[2048, 1]", getitem_25: "i64[2048, 8]", sum_5: "f32[2048, 1]", div_5: "f32[2048, 8]", getitem_27: "i64[16384]", div_6: "i64[16384]", cumsum_2: "i32[128]", unsqueeze_24: "b8[16384, 1]", where_4: "bf16[16384, 2048]", _grouped_mm_2: "bf16[16384, 1536]", mul_29: "bf16[16384, 768]", _grouped_mm_3: "bf16[16384, 2048]", index_put_1: "i64[16384]", add_20: "bf16[4, 512, 2048]", rsqrt_8: "f32[4, 512, 1]", view_44: "bf16[2048, 2048]", mm_10: "bf16[2048, 4096]", rsqrt_9: "f32[4, 512, 32, 1]", mm_11: "bf16[2048, 512]", rsqrt_10: "f32[4, 512, 4, 1]", add_24: "bf16[4, 32, 512, 128]", view_53: "bf16[4, 32, 512, 128]", view_54: "bf16[4, 32, 512, 128]", getitem_30: "bf16[4, 32, 512, 128]", getitem_31: "f32[4, 32, 512, 1]", getitem_36: "i64[]", getitem_37: "i64[]", add_26: "bf16[4, 512, 2048]", rsqrt_11: "f32[4, 512, 1]", view_58: "bf16[2048, 2048]", mm_14: "bf16[2048, 128]", amax_2: "f32[2048, 1]", sum_7: "f32[2048, 1]", getitem_40: "i64[2048, 8]", sum_8: "f32[2048, 1]", div_9: "f32[2048, 8]", getitem_42: "i64[16384]", div_10: "i64[16384]", cumsum_3: "i32[128]", unsqueeze_30: "b8[16384, 1]", where_7: "bf16[16384, 2048]", _grouped_mm_4: "bf16[16384, 1536]", mul_43: "bf16[16384, 768]", _grouped_mm_5: "bf16[16384, 2048]", index_put_2: "i64[16384]", add_29: "bf16[4, 512, 2048]", rsqrt_12: "f32[4, 512, 1]", view_64: "bf16[2048, 2048]", mm_15: "bf16[2048, 4096]", rsqrt_13: "f32[4, 512, 32, 1]", mm_16: "bf16[2048, 512]", rsqrt_14: "f32[4, 512, 4, 1]", add_33: "bf16[4, 32, 512, 128]", view_73: "bf16[4, 32, 512, 128]", view_74: "bf16[4, 32, 512, 128]", getitem_45: "bf16[4, 32, 512, 128]", getitem_46: "f32[4, 32, 512, 1]", getitem_51: "i64[]", getitem_52: "i64[]", add_35: "bf16[4, 512, 2048]", rsqrt_15: "f32[4, 512, 1]", view_78: "bf16[2048, 2048]", mm_19: "bf16[2048, 128]", amax_3: "f32[2048, 1]", sum_10: "f32[2048, 1]", getitem_55: "i64[2048, 8]", sum_11: "f32[2048, 1]", div_13: "f32[2048, 8]", getitem_57: "i64[16384]", div_14: "i64[16384]", cumsum_4: "i32[128]", unsqueeze_36: "b8[16384, 1]", where_10: "bf16[16384, 2048]", _grouped_mm_6: "bf16[16384, 1536]", mul_57: "bf16[16384, 768]", _grouped_mm_7: "bf16[16384, 2048]", index_put_3: "i64[16384]", add_38: "bf16[4, 512, 2048]", rsqrt_16: "f32[4, 512, 1]", view_84: "bf16[2048, 2048]", view_85: "bf16[4, 512, 151936]", constant_pad_nd: "i64[4, 513]", amax_4: "f32[2048, 1]", log: "f32[2048, 1]", convert_element_type_100: "f32[]", tangents_1: "f32[]", tangents_2: "bf16[4, 512, 151936]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_17: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_100);  tangents_1 = convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_20: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_14: "i64[4, 512]" = torch.ops.aten.clone.default(slice_20, memory_format = torch.contiguous_format);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_87: "i64[2048]" = torch.ops.aten.reshape.default(clone_14, [-1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_39: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(view_87, 1);  view_87 = None
        ne_4: "b8[2048, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_39, -100)
        full_default_17: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "i64[2048, 1]" = torch.ops.aten.where.self(ne_4, unsqueeze_39, full_default_17);  unsqueeze_39 = full_default_17 = None

        # No stacktrace found for following nodes
        iota_default: "i64[151936]" = torch.ops.prims.iota.default(151936, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 151936]" = torch.ops.aten.reshape.default(iota_default, [1, 151936]);  iota_default = None
        expand_default: "i64[2048, 151936]" = torch.ops.aten.expand.default(where_14, [2048, 151936]);  where_14 = None
        eq_tensor: "b8[2048, 151936]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[2048, 151936]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f32[2048, 1]" = torch.ops.aten.where.self(ne_4, div_17, full_default_18);  ne_4 = div_17 = None
        mul_61: "f32[2048, 151936]" = torch.ops.aten.mul.Tensor(where_self, where_15);  where_self = where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_99: "f32[4, 512, 151936]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_86: "f32[2048, 151936]" = torch.ops.aten.reshape.default(convert_element_type_99, [-1, 151936]);  convert_element_type_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_6: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(view_86, amax_4);  view_86 = amax_4 = None
        sub_7: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(sub_6, log);  sub_6 = log = None
        exp_9: "f32[2048, 151936]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_16: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_61, [1], True)
        mul_62: "f32[2048, 151936]" = torch.ops.aten.mul.Tensor(exp_9, sum_16);  exp_9 = sum_16 = None
        sub_8: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_88: "f32[4, 512, 151936]" = torch.ops.aten.reshape.default(sub_8, [4, 512, 151936]);  sub_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_101: "bf16[4, 512, 151936]" = torch.ops.prims.convert_element_type.default(view_88, torch.bfloat16);  view_88 = None
        add_40: "bf16[4, 512, 151936]" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_101);  tangents_2 = convert_element_type_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:684 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_89: "bf16[2048, 151936]" = torch.ops.aten.reshape.default(add_40, [2048, 151936]);  add_40 = None
        permute_46: "bf16[151936, 2048]" = torch.ops.aten.permute.default(view_89, [1, 0])
        mm_21: "bf16[151936, 2048]" = torch.ops.aten.mm.default(permute_46, view_84);  permute_46 = view_84 = None
        permute_45: "bf16[2048, 151936]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_48: "bf16[151936, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_22: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_89, permute_48);  view_89 = permute_48 = None
        view_90: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_22, [4, 512, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_63: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_90, primals_48);  primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_95: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_59: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_95, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_96: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        mul_64: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_90, convert_element_type_96);  view_90 = convert_element_type_96 = None
        sum_17: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_64, [0, 1], True);  mul_64 = None
        view_91: "bf16[2048]" = torch.ops.aten.reshape.default(sum_17, [2048]);  sum_17 = None
        convert_element_type_106: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_63, torch.float32);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_65: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_106, convert_element_type_95)
        mul_66: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_106, rsqrt_16);  convert_element_type_106 = None
        sum_18: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_65, [2], True);  mul_65 = None
        pow_18: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_67: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_18, -0.5);  sum_18 = None
        mul_68: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_67, pow_18);  mul_67 = pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_14: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_68, [4, 512, 2048]);  mul_68 = None
        div_18: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_14, 2048);  expand_14 = None
        pow_19: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_95, 1.0);  convert_element_type_95 = None
        mul_69: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_19, 2.0);  pow_19 = None
        mul_70: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_18, mul_69);  div_18 = mul_69 = None
        add_41: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_66, mul_70);  mul_66 = mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_107: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_92: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(convert_element_type_107, [2048, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        unsqueeze_40: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_92, 1);  view_92 = None
        expand_15: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_40, [2048, 8, 2048]);  unsqueeze_40 = None
        clone_15: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_93: "bf16[16384, 2048]" = torch.ops.aten.reshape.default(clone_15, [16384, 2048]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        full_default_22: "bf16[16384, 2048]" = torch.ops.aten.full.default([16384, 2048], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_4: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default_22, [index_put_3], view_93, True);  index_put_3 = view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_16: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default_2, index_put_4);  index_put_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        mul_71: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_16, _grouped_mm_7);  _grouped_mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_91: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_80: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_91, [-1]);  convert_element_type_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_12: "bf16[16384]" = torch.ops.aten.index.Tensor(view_80, [getitem_57]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_37: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_12, -1);  index_12 = None
        mul_72: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_37);  where_16 = unsqueeze_37 = None
        sum_19: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_71, [1], True);  mul_71 = None
        squeeze_1: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_19, -1);  sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_50: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_72, [1, 0])
        _grouped_mm_8: "bf16[128, 2048, 768]" = torch.ops.aten._grouped_mm.default(permute_50, mul_57, cumsum_4);  permute_50 = mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_44: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(primals_47, [0, 2, 1]);  primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_52: "bf16[128, 2048, 768]" = torch.ops.aten.permute.default(permute_44, [0, 2, 1]);  permute_44 = None
        _grouped_mm_9: "bf16[16384, 768]" = torch.ops.aten._grouped_mm.default(mul_72, permute_52, cumsum_4);  mul_72 = permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_3 = torch.ops.aten.split.Tensor(_grouped_mm_6, 768, -1);  _grouped_mm_6 = None
        getitem_58: "bf16[16384, 768]" = split_3[0]
        getitem_59: "bf16[16384, 768]" = split_3[1];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_93: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        neg_11: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_93)
        exp_7: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_37: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_15: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_93, add_37)
        convert_element_type_94: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_73: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_9, convert_element_type_94);  convert_element_type_94 = None
        mul_74: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_9, getitem_59);  _grouped_mm_9 = getitem_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_108: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_74, torch.float32);  mul_74 = None
        reciprocal: "f32[16384, 768]" = torch.ops.aten.reciprocal.default(add_37);  add_37 = None
        mul_75: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_76: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_108, mul_75);  convert_element_type_108 = None
        sub_9: "f32[16384, 768]" = torch.ops.aten.sub.Tensor(1, mul_75);  mul_75 = None
        mul_77: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, sub_9);  convert_element_type_93 = sub_9 = None
        add_43: "f32[16384, 768]" = torch.ops.aten.add.Tensor(mul_77, 1);  mul_77 = None
        mul_78: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(mul_76, add_43);  mul_76 = add_43 = None
        convert_element_type_110: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        cat_9: "bf16[16384, 1536]" = torch.ops.aten.cat.default([convert_element_type_110, mul_73], 1);  convert_element_type_110 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_54: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_9, [1, 0])
        _grouped_mm_10: "bf16[128, 1536, 2048]" = torch.ops.aten._grouped_mm.default(permute_54, where_10, cumsum_4);  permute_54 = where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_43: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(primals_46, [0, 2, 1]);  primals_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_56: "bf16[128, 1536, 2048]" = torch.ops.aten.permute.default(permute_43, [0, 2, 1]);  permute_43 = None
        _grouped_mm_11: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(cat_9, permute_56, cumsum_4);  cat_9 = permute_56 = cumsum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_17: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default_2, _grouped_mm_11);  unsqueeze_36 = _grouped_mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        full_default_25: "bf16[16384]" = torch.ops.aten.full.default([16384], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_5: "bf16[16384]" = torch.ops.aten.index_put.default(full_default_25, [getitem_57], squeeze_1, True);  getitem_57 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        full_default_26: "bf16[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_6: "bf16[2048, 2048]" = torch.ops.aten.index_put.default(full_default_26, [div_14], where_17, True);  div_14 = where_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_94: "bf16[2048, 8]" = torch.ops.aten.reshape.default(index_put_5, [2048, 8]);  index_put_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_111: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(view_94, torch.float32);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        div_20: "f32[2048, 8]" = torch.ops.aten.div.Tensor(div_13, sum_11);  div_13 = None
        neg_14: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_111)
        mul_79: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_14, div_20);  neg_14 = div_20 = None
        div_21: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_111, sum_11);  convert_element_type_111 = sum_11 = None
        sum_20: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_79, [1], True);  mul_79 = None
        expand_16: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_20, [2048, 8]);  sum_20 = None
        add_44: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_21, expand_16);  div_21 = expand_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        full_default_27: "f32[2048, 128]" = torch.ops.aten.full.default([2048, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_1: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_default_27, -1, getitem_55, add_44);  getitem_55 = add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_90: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        sub_5: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_90, amax_3);  convert_element_type_90 = amax_3 = None
        exp_6: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        div_12: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_10);  exp_6 = sum_10 = None
        mul_80: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_1, div_12);  scatter_1 = None
        sum_21: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_80, [-1], True)
        neg_15: "f32[2048, 128]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_15, sum_21, mul_80);  neg_15 = sum_21 = mul_80 = None
        convert_element_type_112: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_58: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0])
        mm_23: "bf16[128, 2048]" = torch.ops.aten.mm.default(permute_58, view_78);  permute_58 = view_78 = None
        permute_42: "bf16[2048, 128]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_60: "bf16[128, 2048]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_24: "bf16[2048, 2048]" = torch.ops.aten.mm.default(convert_element_type_112, permute_60);  convert_element_type_112 = permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:264 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        add_45: "bf16[2048, 2048]" = torch.ops.aten.add.Tensor(index_put_6, mm_24);  index_put_6 = mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_96: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(add_45, [4, 512, 2048]);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_81: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_96, primals_44);  primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_86: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_55: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_86, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_87: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        mul_82: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_96, convert_element_type_87);  view_96 = convert_element_type_87 = None
        sum_22: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_82, [0, 1], True);  mul_82 = None
        view_97: "bf16[2048]" = torch.ops.aten.reshape.default(sum_22, [2048]);  sum_22 = None
        convert_element_type_117: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_81, torch.float32);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_83: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_117, convert_element_type_86)
        mul_84: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_117, rsqrt_15);  convert_element_type_117 = None
        sum_23: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_83, [2], True);  mul_83 = None
        pow_20: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_85: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_23, -0.5);  sum_23 = None
        mul_86: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_85, pow_20);  mul_85 = pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_17: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_86, [4, 512, 2048]);  mul_86 = None
        div_22: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_17, 2048);  expand_17 = None
        pow_21: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_86, 1.0);  convert_element_type_86 = None
        mul_87: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_21, 2.0);  pow_21 = None
        mul_88: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_22, mul_87);  div_22 = mul_87 = None
        add_46: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_84, mul_88);  mul_84 = mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_118: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        add_47: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_107, convert_element_type_118);  convert_element_type_107 = convert_element_type_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_98: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_47, [2048, 2048])
        permute_62: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_98, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_75, [2048, 4096]);  view_75 = None
        mm_25: "bf16[2048, 4096]" = torch.ops.aten.mm.default(permute_62, view_76);  permute_62 = view_76 = None
        permute_41: "bf16[4096, 2048]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_64: "bf16[2048, 4096]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_26: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_98, permute_64);  view_98 = permute_64 = None
        view_99: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 4096]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_100: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_99, [4, 512, 32, 128]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_66, add_33, view_73, view_74, getitem_45, getitem_46, getitem_51, getitem_52, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_66 = add_33 = view_73 = view_74 = getitem_45 = getitem_46 = getitem_51 = getitem_52 = None
        getitem_60: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_61: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_62: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_101: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_62, [4, 4, 8, 512, 128]);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_24: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_101, [2], True);  view_101 = None
        squeeze_2: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_24, 2);  sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_102: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_61, [4, 4, 8, 512, 128]);  getitem_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_25: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_102, [2], True);  view_102 = None
        squeeze_3: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_25, 2);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:493 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:494 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:438 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:439 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_5, torch.float32);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_2, [1, 64, 1]);  expand_2 = None
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:444 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 64]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone, [1, 512, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:446 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_3)
        mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:83 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_89: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_3, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_21: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_89, 3, 0, 64)
        slice_22: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_89, 3, 64, 128);  mul_89 = None
        neg_16: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_21);  slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_28: "bf16[4, 4, 512, 128]" = torch.ops.aten.full.default([4, 4, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, neg_16, 3, 64, 9223372036854775807);  neg_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_1: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_22, 3, 0, 64);  slice_22 = None
        add_48: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:445 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_3);  view_3 = None
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:82 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_90: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_3, unsqueeze_14);  squeeze_3 = None
        add_49: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(add_48, mul_90);  add_48 = mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_91: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_60, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_23: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_91, 3, 0, 64)
        slice_24: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_91, 3, 64, 128);  mul_91 = None
        neg_17: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_23);  slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_30: "bf16[4, 32, 512, 128]" = torch.ops.aten.full.default([4, 32, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, neg_17, 3, 64, 9223372036854775807);  neg_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_3: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_24, 3, 0, 64);  slice_24 = None
        add_50: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_92: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_60, unsqueeze_14);  getitem_60 = None
        add_51: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_50, mul_92);  add_50 = mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(squeeze_2, [0, 2, 1, 3]);  squeeze_2 = None
        clone_16: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_103: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_16, [4, 512, 512]);  clone_16 = None
        view_104: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_103, [2048, 512]);  view_103 = None
        permute_68: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_104, [1, 0])
        mm_27: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_68, view_64);  permute_68 = None
        permute_38: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_70: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_28: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_104, permute_70);  view_104 = permute_70 = None
        view_105: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_28, [4, 512, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_72: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(add_49, [0, 2, 1, 3]);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_93: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_72, primals_41);  primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_68: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 512]);  mm_16 = None
        view_69: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_68, [4, 512, -1, 128]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_80: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_49: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_80, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_81: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        mul_94: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_72, convert_element_type_81);  permute_72 = convert_element_type_81 = None
        sum_26: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_94, [0, 1, 2], True);  mul_94 = None
        view_106: "bf16[128]" = torch.ops.aten.reshape.default(sum_26, [128]);  sum_26 = None
        convert_element_type_127: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_93, torch.float32);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_95: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, convert_element_type_80)
        mul_96: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_14);  convert_element_type_127 = None
        sum_27: "f32[4, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [3], True);  mul_95 = None
        pow_22: "f32[4, 512, 4, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_97: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Scalar(sum_27, -0.5);  sum_27 = None
        mul_98: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Tensor(mul_97, pow_22);  mul_97 = pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_18: "f32[4, 512, 4, 128]" = torch.ops.aten.expand.default(mul_98, [4, 512, 4, 128]);  mul_98 = None
        div_23: "f32[4, 512, 4, 128]" = torch.ops.aten.div.Scalar(expand_18, 128);  expand_18 = None
        pow_23: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_80, 1.0);  convert_element_type_80 = None
        mul_99: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Scalar(pow_23, 2.0);  pow_23 = None
        mul_100: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(div_23, mul_99);  div_23 = mul_99 = None
        add_52: "f32[4, 512, 4, 128]" = torch.ops.aten.add.Tensor(mul_96, mul_100);  mul_96 = mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_128: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_17: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(convert_element_type_128, memory_format = torch.contiguous_format);  convert_element_type_128 = None
        view_107: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_17, [4, 512, 512]);  clone_17 = None
        view_108: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_107, [2048, 512]);  view_107 = None
        permute_73: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_108, [1, 0])
        mm_29: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_73, view_64);  permute_73 = None
        permute_36: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_75: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_30: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_108, permute_75);  view_108 = permute_75 = None
        view_109: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_30, [4, 512, 2048]);  mm_30 = None
        add_53: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_105, view_109);  view_105 = view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_77: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_51, [0, 2, 1, 3]);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_101: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_77, primals_39);  primals_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_65: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 4096]);  mm_15 = None
        view_66: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_65, [4, 512, -1, 128]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_76: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_47: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_76, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_77: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        mul_102: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_77, convert_element_type_77);  permute_77 = convert_element_type_77 = None
        sum_28: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 1, 2], True);  mul_102 = None
        view_110: "bf16[128]" = torch.ops.aten.reshape.default(sum_28, [128]);  sum_28 = None
        convert_element_type_133: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_101, torch.float32);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_103: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_133, convert_element_type_76)
        mul_104: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13);  convert_element_type_133 = None
        sum_29: "f32[4, 512, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [3], True);  mul_103 = None
        pow_24: "f32[4, 512, 32, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_105: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
        mul_106: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Tensor(mul_105, pow_24);  mul_105 = pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_19: "f32[4, 512, 32, 128]" = torch.ops.aten.expand.default(mul_106, [4, 512, 32, 128]);  mul_106 = None
        div_24: "f32[4, 512, 32, 128]" = torch.ops.aten.div.Scalar(expand_19, 128);  expand_19 = None
        pow_25: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_76, 1.0);  convert_element_type_76 = None
        mul_107: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Scalar(pow_25, 2.0);  pow_25 = None
        mul_108: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(div_24, mul_107);  div_24 = mul_107 = None
        add_54: "f32[4, 512, 32, 128]" = torch.ops.aten.add.Tensor(mul_104, mul_108);  mul_104 = mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_134: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_18: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(convert_element_type_134, memory_format = torch.contiguous_format);  convert_element_type_134 = None
        view_111: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_18, [4, 512, 4096]);  clone_18 = None
        view_112: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_111, [2048, 4096]);  view_111 = None
        permute_78: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_112, [1, 0])
        mm_31: "bf16[4096, 2048]" = torch.ops.aten.mm.default(permute_78, view_64);  permute_78 = view_64 = None
        permute_34: "bf16[2048, 4096]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_80: "bf16[4096, 2048]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_32: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_112, permute_80);  view_112 = permute_80 = None
        view_113: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 2048]);  mm_32 = None
        add_55: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_53, view_113);  add_53 = view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_109: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_55, primals_37);  primals_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_72: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_45: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_73: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_110: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_55, convert_element_type_73);  add_55 = convert_element_type_73 = None
        sum_30: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_110, [0, 1], True);  mul_110 = None
        view_114: "bf16[2048]" = torch.ops.aten.reshape.default(sum_30, [2048]);  sum_30 = None
        convert_element_type_139: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_109, torch.float32);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_111: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_139, convert_element_type_72)
        mul_112: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_139, rsqrt_12);  convert_element_type_139 = None
        sum_31: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
        pow_26: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_113: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_31, -0.5);  sum_31 = None
        mul_114: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_113, pow_26);  mul_113 = pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_20: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_114, [4, 512, 2048]);  mul_114 = None
        div_25: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_20, 2048);  expand_20 = None
        pow_27: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_72, 1.0);  convert_element_type_72 = None
        mul_115: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_27, 2.0);  pow_27 = None
        mul_116: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_25, mul_115);  div_25 = mul_115 = None
        add_56: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_112, mul_116);  mul_112 = mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_140: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None
        add_57: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_47, convert_element_type_140);  add_47 = convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_115: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_57, [2048, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        unsqueeze_41: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_115, 1);  view_115 = None
        expand_21: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_41, [2048, 8, 2048]);  unsqueeze_41 = None
        clone_19: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_116: "bf16[16384, 2048]" = torch.ops.aten.reshape.default(clone_19, [16384, 2048]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_put_7: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default_22, [index_put_2], view_116, True);  index_put_2 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_18: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_30, full_default_2, index_put_7);  index_put_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        mul_117: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_18, _grouped_mm_5);  _grouped_mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_68: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_60: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_68, [-1]);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_9: "bf16[16384]" = torch.ops.aten.index.Tensor(view_60, [getitem_42]);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_31: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_9, -1);  index_9 = None
        mul_118: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_31);  where_18 = unsqueeze_31 = None
        sum_32: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [1], True);  mul_117 = None
        squeeze_4: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_32, -1);  sum_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_82: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_118, [1, 0])
        _grouped_mm_12: "bf16[128, 2048, 768]" = torch.ops.aten._grouped_mm.default(permute_82, mul_43, cumsum_3);  permute_82 = mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_33: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(primals_36, [0, 2, 1]);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_84: "bf16[128, 2048, 768]" = torch.ops.aten.permute.default(permute_33, [0, 2, 1]);  permute_33 = None
        _grouped_mm_13: "bf16[16384, 768]" = torch.ops.aten._grouped_mm.default(mul_118, permute_84, cumsum_3);  mul_118 = permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_2 = torch.ops.aten.split.Tensor(_grouped_mm_4, 768, -1);  _grouped_mm_4 = None
        getitem_43: "bf16[16384, 768]" = split_2[0]
        getitem_44: "bf16[16384, 768]" = split_2[1];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_70: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
        neg_8: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_70)
        exp_5: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_28: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_11: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_70, add_28)
        convert_element_type_71: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_119: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_13, convert_element_type_71);  convert_element_type_71 = None
        mul_120: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_13, getitem_44);  _grouped_mm_13 = getitem_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_141: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_120, torch.float32);  mul_120 = None
        reciprocal_1: "f32[16384, 768]" = torch.ops.aten.reciprocal.default(add_28);  add_28 = None
        mul_121: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_122: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_141, mul_121);  convert_element_type_141 = None
        sub_10: "f32[16384, 768]" = torch.ops.aten.sub.Tensor(1, mul_121);  mul_121 = None
        mul_123: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_70, sub_10);  convert_element_type_70 = sub_10 = None
        add_59: "f32[16384, 768]" = torch.ops.aten.add.Tensor(mul_123, 1);  mul_123 = None
        mul_124: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(mul_122, add_59);  mul_122 = add_59 = None
        convert_element_type_143: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_124, torch.bfloat16);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        cat_10: "bf16[16384, 1536]" = torch.ops.aten.cat.default([convert_element_type_143, mul_119], 1);  convert_element_type_143 = mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_86: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_10, [1, 0])
        _grouped_mm_14: "bf16[128, 1536, 2048]" = torch.ops.aten._grouped_mm.default(permute_86, where_7, cumsum_3);  permute_86 = where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_32: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(primals_35, [0, 2, 1]);  primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_88: "bf16[128, 1536, 2048]" = torch.ops.aten.permute.default(permute_32, [0, 2, 1]);  permute_32 = None
        _grouped_mm_15: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(cat_10, permute_88, cumsum_3);  cat_10 = permute_88 = cumsum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_19: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_30, full_default_2, _grouped_mm_15);  unsqueeze_30 = _grouped_mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_put_8: "bf16[16384]" = torch.ops.aten.index_put.default(full_default_25, [getitem_42], squeeze_4, True);  getitem_42 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        index_put_9: "bf16[2048, 2048]" = torch.ops.aten.index_put.default(full_default_26, [div_10], where_19, True);  div_10 = where_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_117: "bf16[2048, 8]" = torch.ops.aten.reshape.default(index_put_8, [2048, 8]);  index_put_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_144: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        div_27: "f32[2048, 8]" = torch.ops.aten.div.Tensor(div_9, sum_8);  div_9 = None
        neg_19: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_144)
        mul_125: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_19, div_27);  neg_19 = div_27 = None
        div_28: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_144, sum_8);  convert_element_type_144 = sum_8 = None
        sum_33: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [1], True);  mul_125 = None
        expand_22: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_33, [2048, 8]);  sum_33 = None
        add_60: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_28, expand_22);  div_28 = expand_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        scatter_2: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_default_27, -1, getitem_40, add_60);  getitem_40 = add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_67: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        sub_4: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_67, amax_2);  convert_element_type_67 = amax_2 = None
        exp_4: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_8: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_7);  exp_4 = sum_7 = None
        mul_126: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_2, div_8);  scatter_2 = None
        sum_34: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [-1], True)
        neg_20: "f32[2048, 128]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_1: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_20, sum_34, mul_126);  neg_20 = sum_34 = mul_126 = None
        convert_element_type_145: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_90: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0])
        mm_33: "bf16[128, 2048]" = torch.ops.aten.mm.default(permute_90, view_58);  permute_90 = view_58 = None
        permute_31: "bf16[2048, 128]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_92: "bf16[128, 2048]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_34: "bf16[2048, 2048]" = torch.ops.aten.mm.default(convert_element_type_145, permute_92);  convert_element_type_145 = permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:264 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        add_61: "bf16[2048, 2048]" = torch.ops.aten.add.Tensor(index_put_9, mm_34);  index_put_9 = mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_119: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(add_61, [4, 512, 2048]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_127: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_119, primals_33);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_63: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_41: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_64: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        mul_128: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_119, convert_element_type_64);  view_119 = convert_element_type_64 = None
        sum_35: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1], True);  mul_128 = None
        view_120: "bf16[2048]" = torch.ops.aten.reshape.default(sum_35, [2048]);  sum_35 = None
        convert_element_type_150: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_127, torch.float32);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_129: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_150, convert_element_type_63)
        mul_130: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_150, rsqrt_11);  convert_element_type_150 = None
        sum_36: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
        pow_28: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_131: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_36, -0.5);  sum_36 = None
        mul_132: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_131, pow_28);  mul_131 = pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_23: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_132, [4, 512, 2048]);  mul_132 = None
        div_29: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_23, 2048);  expand_23 = None
        pow_29: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 1.0);  convert_element_type_63 = None
        mul_133: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_29, 2.0);  pow_29 = None
        mul_134: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_29, mul_133);  div_29 = mul_133 = None
        add_62: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_151: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        add_63: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_57, convert_element_type_151);  add_57 = convert_element_type_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_121: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_63, [2048, 2048])
        permute_94: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_121, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_30, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_55, [2048, 4096]);  view_55 = None
        mm_35: "bf16[2048, 4096]" = torch.ops.aten.mm.default(permute_94, view_56);  permute_94 = view_56 = None
        permute_30: "bf16[4096, 2048]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_96: "bf16[2048, 4096]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_36: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_121, permute_96);  view_121 = permute_96 = None
        view_122: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_36, [4, 512, 4096]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_123: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_122, [4, 512, 32, 128]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_98: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_98, add_24, view_53, view_54, getitem_30, getitem_31, getitem_36, getitem_37, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_98 = add_24 = view_53 = view_54 = getitem_30 = getitem_31 = getitem_36 = getitem_37 = None
        getitem_63: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_64: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_65: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_124: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_65, [4, 4, 8, 512, 128]);  getitem_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_37: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_124, [2], True);  view_124 = None
        squeeze_5: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_37, 2);  sum_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_125: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_64, [4, 4, 8, 512, 128]);  getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_38: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_125, [2], True);  view_125 = None
        squeeze_6: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_38, 2);  sum_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_135: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_6, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_25: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_135, 3, 0, 64)
        slice_26: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_135, 3, 64, 128);  mul_135 = None
        neg_21: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_4: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, neg_21, 3, 64, 9223372036854775807);  neg_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_5: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_26, 3, 0, 64);  slice_26 = None
        add_64: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_136: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_6, unsqueeze_14);  squeeze_6 = None
        add_65: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(add_64, mul_136);  add_64 = mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_137: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_63, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_27: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_137, 3, 0, 64)
        slice_28: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_137, 3, 64, 128);  mul_137 = None
        neg_22: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_27);  slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_6: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, neg_22, 3, 64, 9223372036854775807);  neg_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_7: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_28, 3, 0, 64);  slice_28 = None
        add_66: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_138: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_63, unsqueeze_14);  getitem_63 = None
        add_67: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_66, mul_138);  add_66 = mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_99: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(squeeze_5, [0, 2, 1, 3]);  squeeze_5 = None
        clone_20: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_126: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_20, [4, 512, 512]);  clone_20 = None
        view_127: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_126, [2048, 512]);  view_126 = None
        permute_100: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_127, [1, 0])
        mm_37: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_100, view_44);  permute_100 = None
        permute_27: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_102: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_38: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_127, permute_102);  view_127 = permute_102 = None
        view_128: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_38, [4, 512, 2048]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_104: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_139: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_104, primals_30);  primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_48: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 512]);  mm_11 = None
        view_49: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_48, [4, 512, -1, 128]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_57: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_35: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_57, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_58: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        mul_140: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_104, convert_element_type_58);  permute_104 = convert_element_type_58 = None
        sum_39: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1, 2], True);  mul_140 = None
        view_129: "bf16[128]" = torch.ops.aten.reshape.default(sum_39, [128]);  sum_39 = None
        convert_element_type_160: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_139, torch.float32);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_141: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_160, convert_element_type_57)
        mul_142: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_160, rsqrt_10);  convert_element_type_160 = None
        sum_40: "f32[4, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [3], True);  mul_141 = None
        pow_30: "f32[4, 512, 4, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_143: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
        mul_144: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Tensor(mul_143, pow_30);  mul_143 = pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_24: "f32[4, 512, 4, 128]" = torch.ops.aten.expand.default(mul_144, [4, 512, 4, 128]);  mul_144 = None
        div_30: "f32[4, 512, 4, 128]" = torch.ops.aten.div.Scalar(expand_24, 128);  expand_24 = None
        pow_31: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_57, 1.0);  convert_element_type_57 = None
        mul_145: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Scalar(pow_31, 2.0);  pow_31 = None
        mul_146: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(div_30, mul_145);  div_30 = mul_145 = None
        add_68: "f32[4, 512, 4, 128]" = torch.ops.aten.add.Tensor(mul_142, mul_146);  mul_142 = mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_161: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_21: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(convert_element_type_161, memory_format = torch.contiguous_format);  convert_element_type_161 = None
        view_130: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_21, [4, 512, 512]);  clone_21 = None
        view_131: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_130, [2048, 512]);  view_130 = None
        permute_105: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_39: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_105, view_44);  permute_105 = None
        permute_25: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_107: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_40: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_131, permute_107);  view_131 = permute_107 = None
        view_132: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 2048]);  mm_40 = None
        add_69: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_128, view_132);  view_128 = view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_109: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_67, [0, 2, 1, 3]);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_147: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_109, primals_28);  primals_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_45: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 4096]);  mm_10 = None
        view_46: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_45, [4, 512, -1, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_53: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_33: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_54: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        mul_148: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_109, convert_element_type_54);  permute_109 = convert_element_type_54 = None
        sum_41: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1, 2], True);  mul_148 = None
        view_133: "bf16[128]" = torch.ops.aten.reshape.default(sum_41, [128]);  sum_41 = None
        convert_element_type_166: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_147, torch.float32);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_149: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_166, convert_element_type_53)
        mul_150: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_166, rsqrt_9);  convert_element_type_166 = None
        sum_42: "f32[4, 512, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [3], True);  mul_149 = None
        pow_32: "f32[4, 512, 32, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_151: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
        mul_152: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Tensor(mul_151, pow_32);  mul_151 = pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_25: "f32[4, 512, 32, 128]" = torch.ops.aten.expand.default(mul_152, [4, 512, 32, 128]);  mul_152 = None
        div_31: "f32[4, 512, 32, 128]" = torch.ops.aten.div.Scalar(expand_25, 128);  expand_25 = None
        pow_33: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 1.0);  convert_element_type_53 = None
        mul_153: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Scalar(pow_33, 2.0);  pow_33 = None
        mul_154: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(div_31, mul_153);  div_31 = mul_153 = None
        add_70: "f32[4, 512, 32, 128]" = torch.ops.aten.add.Tensor(mul_150, mul_154);  mul_150 = mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_167: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_22: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(convert_element_type_167, memory_format = torch.contiguous_format);  convert_element_type_167 = None
        view_134: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_22, [4, 512, 4096]);  clone_22 = None
        view_135: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_134, [2048, 4096]);  view_134 = None
        permute_110: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_41: "bf16[4096, 2048]" = torch.ops.aten.mm.default(permute_110, view_44);  permute_110 = view_44 = None
        permute_23: "bf16[2048, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_112: "bf16[4096, 2048]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_42: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_135, permute_112);  view_135 = permute_112 = None
        view_136: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_42, [4, 512, 2048]);  mm_42 = None
        add_71: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_69, view_136);  add_69 = view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_155: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_71, primals_26);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_49: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_31: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_49, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_50: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        mul_156: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_71, convert_element_type_50);  add_71 = convert_element_type_50 = None
        sum_43: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 1], True);  mul_156 = None
        view_137: "bf16[2048]" = torch.ops.aten.reshape.default(sum_43, [2048]);  sum_43 = None
        convert_element_type_172: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_155, torch.float32);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_157: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_172, convert_element_type_49)
        mul_158: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_172, rsqrt_8);  convert_element_type_172 = None
        sum_44: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
        pow_34: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_159: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_44, -0.5);  sum_44 = None
        mul_160: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_159, pow_34);  mul_159 = pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_26: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_160, [4, 512, 2048]);  mul_160 = None
        div_32: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_26, 2048);  expand_26 = None
        pow_35: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_49, 1.0);  convert_element_type_49 = None
        mul_161: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_35, 2.0);  pow_35 = None
        mul_162: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_32, mul_161);  div_32 = mul_161 = None
        add_72: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_158, mul_162);  mul_158 = mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_173: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        add_73: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_63, convert_element_type_173);  add_63 = convert_element_type_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_138: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_73, [2048, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        unsqueeze_42: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_138, 1);  view_138 = None
        expand_27: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_42, [2048, 8, 2048]);  unsqueeze_42 = None
        clone_23: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_139: "bf16[16384, 2048]" = torch.ops.aten.reshape.default(clone_23, [16384, 2048]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_put_10: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default_22, [index_put_1], view_139, True);  index_put_1 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_20: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_24, full_default_2, index_put_10);  index_put_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        mul_163: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_20, _grouped_mm_3);  _grouped_mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_45: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_40: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_45, [-1]);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_6: "bf16[16384]" = torch.ops.aten.index.Tensor(view_40, [getitem_27]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_25: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_6, -1);  index_6 = None
        mul_164: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_25);  where_20 = unsqueeze_25 = None
        sum_45: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [1], True);  mul_163 = None
        squeeze_7: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_45, -1);  sum_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_114: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_164, [1, 0])
        _grouped_mm_16: "bf16[128, 2048, 768]" = torch.ops.aten._grouped_mm.default(permute_114, mul_29, cumsum_2);  permute_114 = mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_22: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(primals_25, [0, 2, 1]);  primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_116: "bf16[128, 2048, 768]" = torch.ops.aten.permute.default(permute_22, [0, 2, 1]);  permute_22 = None
        _grouped_mm_17: "bf16[16384, 768]" = torch.ops.aten._grouped_mm.default(mul_164, permute_116, cumsum_2);  mul_164 = permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_1 = torch.ops.aten.split.Tensor(_grouped_mm_2, 768, -1);  _grouped_mm_2 = None
        getitem_28: "bf16[16384, 768]" = split_1[0]
        getitem_29: "bf16[16384, 768]" = split_1[1];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_47: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        neg_5: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_47)
        exp_3: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_19: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_7: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_47, add_19)
        convert_element_type_48: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_165: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_17, convert_element_type_48);  convert_element_type_48 = None
        mul_166: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_17, getitem_29);  _grouped_mm_17 = getitem_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_174: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_166, torch.float32);  mul_166 = None
        reciprocal_2: "f32[16384, 768]" = torch.ops.aten.reciprocal.default(add_19);  add_19 = None
        mul_167: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_168: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_174, mul_167);  convert_element_type_174 = None
        sub_11: "f32[16384, 768]" = torch.ops.aten.sub.Tensor(1, mul_167);  mul_167 = None
        mul_169: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_47, sub_11);  convert_element_type_47 = sub_11 = None
        add_75: "f32[16384, 768]" = torch.ops.aten.add.Tensor(mul_169, 1);  mul_169 = None
        mul_170: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(mul_168, add_75);  mul_168 = add_75 = None
        convert_element_type_176: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        cat_11: "bf16[16384, 1536]" = torch.ops.aten.cat.default([convert_element_type_176, mul_165], 1);  convert_element_type_176 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_118: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_11, [1, 0])
        _grouped_mm_18: "bf16[128, 1536, 2048]" = torch.ops.aten._grouped_mm.default(permute_118, where_4, cumsum_2);  permute_118 = where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_21: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(primals_24, [0, 2, 1]);  primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_120: "bf16[128, 1536, 2048]" = torch.ops.aten.permute.default(permute_21, [0, 2, 1]);  permute_21 = None
        _grouped_mm_19: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(cat_11, permute_120, cumsum_2);  cat_11 = permute_120 = cumsum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_21: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_24, full_default_2, _grouped_mm_19);  unsqueeze_24 = _grouped_mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_put_11: "bf16[16384]" = torch.ops.aten.index_put.default(full_default_25, [getitem_27], squeeze_7, True);  getitem_27 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        index_put_12: "bf16[2048, 2048]" = torch.ops.aten.index_put.default(full_default_26, [div_6], where_21, True);  div_6 = where_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_140: "bf16[2048, 8]" = torch.ops.aten.reshape.default(index_put_11, [2048, 8]);  index_put_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_177: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(view_140, torch.float32);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        div_34: "f32[2048, 8]" = torch.ops.aten.div.Tensor(div_5, sum_5);  div_5 = None
        neg_24: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_177)
        mul_171: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_24, div_34);  neg_24 = div_34 = None
        div_35: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_177, sum_5);  convert_element_type_177 = sum_5 = None
        sum_46: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [1], True);  mul_171 = None
        expand_28: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_46, [2048, 8]);  sum_46 = None
        add_76: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_35, expand_28);  div_35 = expand_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        scatter_3: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_default_27, -1, getitem_25, add_76);  getitem_25 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_44: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        sub_3: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_44, amax_1);  convert_element_type_44 = amax_1 = None
        exp_2: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        div_4: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_4);  exp_2 = sum_4 = None
        mul_172: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_3, div_4);  scatter_3 = None
        sum_47: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [-1], True)
        neg_25: "f32[2048, 128]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_2: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_25, sum_47, mul_172);  neg_25 = sum_47 = mul_172 = None
        convert_element_type_178: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_122: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0])
        mm_43: "bf16[128, 2048]" = torch.ops.aten.mm.default(permute_122, view_38);  permute_122 = view_38 = None
        permute_20: "bf16[2048, 128]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_124: "bf16[128, 2048]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_44: "bf16[2048, 2048]" = torch.ops.aten.mm.default(convert_element_type_178, permute_124);  convert_element_type_178 = permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:264 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        add_77: "bf16[2048, 2048]" = torch.ops.aten.add.Tensor(index_put_12, mm_44);  index_put_12 = mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_142: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(add_77, [4, 512, 2048]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_173: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_142, primals_22);  primals_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_40: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_27: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_40, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_41: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_174: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_142, convert_element_type_41);  view_142 = convert_element_type_41 = None
        sum_48: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1], True);  mul_174 = None
        view_143: "bf16[2048]" = torch.ops.aten.reshape.default(sum_48, [2048]);  sum_48 = None
        convert_element_type_183: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_173, torch.float32);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_175: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_183, convert_element_type_40)
        mul_176: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_183, rsqrt_7);  convert_element_type_183 = None
        sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
        pow_36: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_177: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_49, -0.5);  sum_49 = None
        mul_178: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_177, pow_36);  mul_177 = pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_29: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_178, [4, 512, 2048]);  mul_178 = None
        div_36: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_29, 2048);  expand_29 = None
        pow_37: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_40, 1.0);  convert_element_type_40 = None
        mul_179: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
        mul_180: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_36, mul_179);  div_36 = mul_179 = None
        add_78: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_176, mul_180);  mul_176 = mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_184: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        add_79: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_184);  add_73 = convert_element_type_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_144: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_79, [2048, 2048])
        permute_126: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_144, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_35, [2048, 4096]);  view_35 = None
        mm_45: "bf16[2048, 4096]" = torch.ops.aten.mm.default(permute_126, view_36);  permute_126 = view_36 = None
        permute_19: "bf16[4096, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_128: "bf16[2048, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_46: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_144, permute_128);  view_144 = permute_128 = None
        view_145: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 4096]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_146: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_145, [4, 512, 32, 128]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_130: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_130, add_15, view_33, view_34, getitem_15, getitem_16, getitem_21, getitem_22, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_130 = add_15 = view_33 = view_34 = getitem_15 = getitem_16 = getitem_21 = getitem_22 = None
        getitem_66: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_67: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_68: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_147: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_68, [4, 4, 8, 512, 128]);  getitem_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_50: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_147, [2], True);  view_147 = None
        squeeze_8: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_50, 2);  sum_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_148: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_67, [4, 4, 8, 512, 128]);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_51: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_148, [2], True);  view_148 = None
        squeeze_9: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_51, 2);  sum_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_181: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_9, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_29: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_181, 3, 0, 64)
        slice_30: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_181, 3, 64, 128);  mul_181 = None
        neg_26: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_29);  slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_8: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, neg_26, 3, 64, 9223372036854775807);  neg_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_9: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_30, 3, 0, 64);  slice_30 = None
        add_80: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_182: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_9, unsqueeze_14);  squeeze_9 = None
        add_81: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(add_80, mul_182);  add_80 = mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_183: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_66, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_31: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_183, 3, 0, 64)
        slice_32: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_183, 3, 64, 128);  mul_183 = None
        neg_27: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_31);  slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_10: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, neg_27, 3, 64, 9223372036854775807);  neg_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_11: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_32, 3, 0, 64);  slice_32 = None
        add_82: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_184: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_66, unsqueeze_14);  getitem_66 = None
        add_83: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_82, mul_184);  add_82 = mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_131: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(squeeze_8, [0, 2, 1, 3]);  squeeze_8 = None
        clone_24: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        view_149: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_24, [4, 512, 512]);  clone_24 = None
        view_150: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_149, [2048, 512]);  view_149 = None
        permute_132: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_47: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_132, view_24);  permute_132 = None
        permute_16: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_134: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_48: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_150, permute_134);  view_150 = permute_134 = None
        view_151: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 2048]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_136: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(add_81, [0, 2, 1, 3]);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_185: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_136, primals_19);  primals_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_28: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 512]);  mm_6 = None
        view_29: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_28, [4, 512, -1, 128]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_34: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_21: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_35: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_186: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_136, convert_element_type_35);  permute_136 = convert_element_type_35 = None
        sum_52: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1, 2], True);  mul_186 = None
        view_152: "bf16[128]" = torch.ops.aten.reshape.default(sum_52, [128]);  sum_52 = None
        convert_element_type_193: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_185, torch.float32);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_187: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_193, convert_element_type_34)
        mul_188: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_193, rsqrt_6);  convert_element_type_193 = None
        sum_53: "f32[4, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [3], True);  mul_187 = None
        pow_38: "f32[4, 512, 4, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_189: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Scalar(sum_53, -0.5);  sum_53 = None
        mul_190: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Tensor(mul_189, pow_38);  mul_189 = pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_30: "f32[4, 512, 4, 128]" = torch.ops.aten.expand.default(mul_190, [4, 512, 4, 128]);  mul_190 = None
        div_37: "f32[4, 512, 4, 128]" = torch.ops.aten.div.Scalar(expand_30, 128);  expand_30 = None
        pow_39: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 1.0);  convert_element_type_34 = None
        mul_191: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
        mul_192: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(div_37, mul_191);  div_37 = mul_191 = None
        add_84: "f32[4, 512, 4, 128]" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_194: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_25: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(convert_element_type_194, memory_format = torch.contiguous_format);  convert_element_type_194 = None
        view_153: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_25, [4, 512, 512]);  clone_25 = None
        view_154: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_153, [2048, 512]);  view_153 = None
        permute_137: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_49: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_137, view_24);  permute_137 = None
        permute_14: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_139: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_50: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_154, permute_139);  view_154 = permute_139 = None
        view_155: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 2048]);  mm_50 = None
        add_85: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_151, view_155);  view_151 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_141: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_83, [0, 2, 1, 3]);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_193: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_141, primals_17);  primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_25: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 4096]);  mm_5 = None
        view_26: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_25, [4, 512, -1, 128]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_30: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_19: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_30, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_31: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        mul_194: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_141, convert_element_type_31);  permute_141 = convert_element_type_31 = None
        sum_54: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 1, 2], True);  mul_194 = None
        view_156: "bf16[128]" = torch.ops.aten.reshape.default(sum_54, [128]);  sum_54 = None
        convert_element_type_199: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_193, torch.float32);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_195: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_199, convert_element_type_30)
        mul_196: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_199, rsqrt_5);  convert_element_type_199 = None
        sum_55: "f32[4, 512, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [3], True);  mul_195 = None
        pow_40: "f32[4, 512, 32, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_197: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Scalar(sum_55, -0.5);  sum_55 = None
        mul_198: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Tensor(mul_197, pow_40);  mul_197 = pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_31: "f32[4, 512, 32, 128]" = torch.ops.aten.expand.default(mul_198, [4, 512, 32, 128]);  mul_198 = None
        div_38: "f32[4, 512, 32, 128]" = torch.ops.aten.div.Scalar(expand_31, 128);  expand_31 = None
        pow_41: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_30, 1.0);  convert_element_type_30 = None
        mul_199: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
        mul_200: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(div_38, mul_199);  div_38 = mul_199 = None
        add_86: "f32[4, 512, 32, 128]" = torch.ops.aten.add.Tensor(mul_196, mul_200);  mul_196 = mul_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_200: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_26: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(convert_element_type_200, memory_format = torch.contiguous_format);  convert_element_type_200 = None
        view_157: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_26, [4, 512, 4096]);  clone_26 = None
        view_158: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_157, [2048, 4096]);  view_157 = None
        permute_142: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_158, [1, 0])
        mm_51: "bf16[4096, 2048]" = torch.ops.aten.mm.default(permute_142, view_24);  permute_142 = view_24 = None
        permute_12: "bf16[2048, 4096]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_144: "bf16[4096, 2048]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_52: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_158, permute_144);  view_158 = permute_144 = None
        view_159: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 2048]);  mm_52 = None
        add_87: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_85, view_159);  add_85 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_201: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_87, primals_15);  primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_26: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_17: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_26, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_27: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        mul_202: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_87, convert_element_type_27);  add_87 = convert_element_type_27 = None
        sum_56: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1], True);  mul_202 = None
        view_160: "bf16[2048]" = torch.ops.aten.reshape.default(sum_56, [2048]);  sum_56 = None
        convert_element_type_205: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_201, torch.float32);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_203: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_205, convert_element_type_26)
        mul_204: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_205, rsqrt_4);  convert_element_type_205 = None
        sum_57: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
        pow_42: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_205: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_57, -0.5);  sum_57 = None
        mul_206: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_205, pow_42);  mul_205 = pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_32: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_206, [4, 512, 2048]);  mul_206 = None
        div_39: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_32, 2048);  expand_32 = None
        pow_43: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_26, 1.0);  convert_element_type_26 = None
        mul_207: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
        mul_208: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_39, mul_207);  div_39 = mul_207 = None
        add_88: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_204, mul_208);  mul_204 = mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_206: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None
        add_89: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_206);  add_79 = convert_element_type_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_161: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_89, [2048, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        unsqueeze_43: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_161, 1);  view_161 = None
        expand_33: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_43, [2048, 8, 2048]);  unsqueeze_43 = None
        clone_27: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_162: "bf16[16384, 2048]" = torch.ops.aten.reshape.default(clone_27, [16384, 2048]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_put_13: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default_22, [index_put], view_162, True);  full_default_22 = index_put = view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_22: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_18, full_default_2, index_put_13);  index_put_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        mul_209: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_22, _grouped_mm_1);  _grouped_mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_22: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_20: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_22, [-1]);  convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_3: "bf16[16384]" = torch.ops.aten.index.Tensor(view_20, [getitem_12]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_19: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_3, -1);  index_3 = None
        mul_210: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_19);  where_22 = unsqueeze_19 = None
        sum_58: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [1], True);  mul_209 = None
        squeeze_10: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_58, -1);  sum_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_146: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_210, [1, 0])
        _grouped_mm_20: "bf16[128, 2048, 768]" = torch.ops.aten._grouped_mm.default(permute_146, mul_15, cumsum_1);  permute_146 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_11: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(primals_14, [0, 2, 1]);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_148: "bf16[128, 2048, 768]" = torch.ops.aten.permute.default(permute_11, [0, 2, 1]);  permute_11 = None
        _grouped_mm_21: "bf16[16384, 768]" = torch.ops.aten._grouped_mm.default(mul_210, permute_148, cumsum_1);  mul_210 = permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split = torch.ops.aten.split.Tensor(_grouped_mm, 768, -1);  _grouped_mm = None
        getitem_13: "bf16[16384, 768]" = split[0]
        getitem_14: "bf16[16384, 768]" = split[1];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_24: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        neg_2: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_24)
        exp_1: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_10: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_3: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_24, add_10)
        convert_element_type_25: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_211: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_21, convert_element_type_25);  convert_element_type_25 = None
        mul_212: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_21, getitem_14);  _grouped_mm_21 = getitem_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_207: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_212, torch.float32);  mul_212 = None
        reciprocal_3: "f32[16384, 768]" = torch.ops.aten.reciprocal.default(add_10);  add_10 = None
        mul_213: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_214: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_207, mul_213);  convert_element_type_207 = None
        sub_12: "f32[16384, 768]" = torch.ops.aten.sub.Tensor(1, mul_213);  mul_213 = None
        mul_215: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, sub_12);  convert_element_type_24 = sub_12 = None
        add_91: "f32[16384, 768]" = torch.ops.aten.add.Tensor(mul_215, 1);  mul_215 = None
        mul_216: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(mul_214, add_91);  mul_214 = add_91 = None
        convert_element_type_209: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        cat_12: "bf16[16384, 1536]" = torch.ops.aten.cat.default([convert_element_type_209, mul_211], 1);  convert_element_type_209 = mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_150: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_12, [1, 0])
        _grouped_mm_22: "bf16[128, 1536, 2048]" = torch.ops.aten._grouped_mm.default(permute_150, where_1, cumsum_1);  permute_150 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_10: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(primals_13, [0, 2, 1]);  primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_152: "bf16[128, 1536, 2048]" = torch.ops.aten.permute.default(permute_10, [0, 2, 1]);  permute_10 = None
        _grouped_mm_23: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(cat_12, permute_152, cumsum_1);  cat_12 = permute_152 = cumsum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_23: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_18, full_default_2, _grouped_mm_23);  unsqueeze_18 = full_default_2 = _grouped_mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_put_14: "bf16[16384]" = torch.ops.aten.index_put.default(full_default_25, [getitem_12], squeeze_10, True);  full_default_25 = getitem_12 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        index_put_15: "bf16[2048, 2048]" = torch.ops.aten.index_put.default(full_default_26, [div_2], where_23, True);  full_default_26 = div_2 = where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_163: "bf16[2048, 8]" = torch.ops.aten.reshape.default(index_put_14, [2048, 8]);  index_put_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_210: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(view_163, torch.float32);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        div_41: "f32[2048, 8]" = torch.ops.aten.div.Tensor(div_1, sum_2);  div_1 = None
        neg_29: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_210)
        mul_217: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_29, div_41);  neg_29 = div_41 = None
        div_42: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_210, sum_2);  convert_element_type_210 = sum_2 = None
        sum_59: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [1], True);  mul_217 = None
        expand_34: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_59, [2048, 8]);  sum_59 = None
        add_92: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_42, expand_34);  div_42 = expand_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        scatter_4: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_default_27, -1, getitem_10, add_92);  full_default_27 = getitem_10 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_21: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        sub_2: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_21, amax);  convert_element_type_21 = amax = None
        exp: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_218: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_4, div);  scatter_4 = None
        sum_60: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [-1], True)
        neg_30: "f32[2048, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma_3: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_30, sum_60, mul_218);  neg_30 = sum_60 = mul_218 = None
        convert_element_type_211: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_154: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0])
        mm_53: "bf16[128, 2048]" = torch.ops.aten.mm.default(permute_154, view_18);  permute_154 = view_18 = None
        permute_9: "bf16[2048, 128]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_156: "bf16[128, 2048]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_54: "bf16[2048, 2048]" = torch.ops.aten.mm.default(convert_element_type_211, permute_156);  convert_element_type_211 = permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:264 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        add_93: "bf16[2048, 2048]" = torch.ops.aten.add.Tensor(index_put_15, mm_54);  index_put_15 = mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_165: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(add_93, [4, 512, 2048]);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_219: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_165, primals_11);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_17: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 2048]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:346 in forward, code: hidden_states = residual + hidden_states
        add_8: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(embedding, view_17);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_17: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_13: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_18: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        mul_220: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_165, convert_element_type_18);  view_165 = convert_element_type_18 = None
        sum_61: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True);  mul_220 = None
        view_166: "bf16[2048]" = torch.ops.aten.reshape.default(sum_61, [2048]);  sum_61 = None
        convert_element_type_216: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_219, torch.float32);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_221: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_216, convert_element_type_17)
        mul_222: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_216, rsqrt_3);  convert_element_type_216 = None
        sum_62: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True);  mul_221 = None
        pow_44: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_223: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_62, -0.5);  sum_62 = None
        mul_224: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_223, pow_44);  mul_223 = pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_35: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_224, [4, 512, 2048]);  mul_224 = None
        div_43: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_35, 2048);  expand_35 = None
        pow_45: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 1.0);  convert_element_type_17 = None
        mul_225: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
        mul_226: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_43, mul_225);  div_43 = mul_225 = None
        add_94: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_222, mul_226);  mul_222 = mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_217: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        add_95: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_89, convert_element_type_217);  add_89 = convert_element_type_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_167: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_95, [2048, 2048])
        permute_158: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_167, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_15, [2048, 4096]);  view_15 = None
        mm_55: "bf16[2048, 4096]" = torch.ops.aten.mm.default(permute_158, view_16);  permute_158 = view_16 = None
        permute_8: "bf16[4096, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_160: "bf16[2048, 4096]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_56: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_167, permute_160);  view_167 = permute_160 = None
        view_168: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_56, [4, 512, 4096]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_168, [4, 512, 32, 128]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_162: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_162, add_6, view_13, view_14, getitem, getitem_1, getitem_6, getitem_7, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_162 = add_6 = view_13 = view_14 = getitem = getitem_1 = getitem_6 = getitem_7 = where = None
        getitem_69: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_70: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_71: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_170: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_71, [4, 4, 8, 512, 128]);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_63: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_170, [2], True);  view_170 = None
        squeeze_11: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_63, 2);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_171: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.reshape.default(getitem_70, [4, 4, 8, 512, 128]);  getitem_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_64: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_171, [2], True);  view_171 = None
        squeeze_12: "bf16[4, 4, 512, 128]" = torch.ops.aten.squeeze.dim(sum_64, 2);  sum_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_227: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_12, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_33: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_227, 3, 0, 64)
        slice_34: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(mul_227, 3, 64, 128);  mul_227 = None
        neg_31: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_33);  slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_12: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, neg_31, 3, 64, 9223372036854775807);  neg_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_13: "bf16[4, 4, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_34, 3, 0, 64);  full_default_28 = slice_34 = None
        add_96: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_228: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_12, unsqueeze_14);  squeeze_12 = None
        add_97: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(add_96, mul_228);  add_96 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_229: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_69, unsqueeze_15);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_35: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_229, 3, 0, 64)
        slice_36: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_229, 3, 64, 128);  mul_229 = None
        neg_32: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_35);  slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_14: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, neg_32, 3, 64, 9223372036854775807);  neg_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_15: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_36, 3, 0, 64);  full_default_30 = slice_36 = None
        add_98: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_230: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_69, unsqueeze_14);  getitem_69 = unsqueeze_14 = None
        add_99: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_98, mul_230);  add_98 = mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_163: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(squeeze_11, [0, 2, 1, 3]);  squeeze_11 = None
        clone_28: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        view_172: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_28, [4, 512, 512]);  clone_28 = None
        view_173: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_172, [2048, 512]);  view_172 = None
        permute_164: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_57: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_164, view_4);  permute_164 = None
        permute_5: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_166: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_58: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_173, permute_166);  view_173 = permute_166 = None
        view_174: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 2048]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_168: "bf16[4, 512, 4, 128]" = torch.ops.aten.permute.default(add_97, [0, 2, 1, 3]);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_231: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_168, primals_8);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_8: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 512]);  mm_1 = None
        view_9: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_8, [4, 512, -1, 128]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_11: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_7: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_12: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        mul_232: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_168, convert_element_type_12);  permute_168 = convert_element_type_12 = None
        sum_65: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1, 2], True);  mul_232 = None
        view_175: "bf16[128]" = torch.ops.aten.reshape.default(sum_65, [128]);  sum_65 = None
        convert_element_type_226: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_231, torch.float32);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_233: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_226, convert_element_type_11)
        mul_234: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_226, rsqrt_2);  convert_element_type_226 = None
        sum_66: "f32[4, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [3], True);  mul_233 = None
        pow_46: "f32[4, 512, 4, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_235: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
        mul_236: "f32[4, 512, 4, 1]" = torch.ops.aten.mul.Tensor(mul_235, pow_46);  mul_235 = pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_36: "f32[4, 512, 4, 128]" = torch.ops.aten.expand.default(mul_236, [4, 512, 4, 128]);  mul_236 = None
        div_44: "f32[4, 512, 4, 128]" = torch.ops.aten.div.Scalar(expand_36, 128);  expand_36 = None
        pow_47: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 1.0);  convert_element_type_11 = None
        mul_237: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
        mul_238: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(div_44, mul_237);  div_44 = mul_237 = None
        add_100: "f32[4, 512, 4, 128]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_227: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_29: "bf16[4, 512, 4, 128]" = torch.ops.aten.clone.default(convert_element_type_227, memory_format = torch.contiguous_format);  convert_element_type_227 = None
        view_176: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_29, [4, 512, 512]);  clone_29 = None
        view_177: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_176, [2048, 512]);  view_176 = None
        permute_169: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_59: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_169, view_4);  permute_169 = None
        permute_3: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_171: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_60: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_177, permute_171);  view_177 = permute_171 = None
        view_178: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 2048]);  mm_60 = None
        add_101: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_174, view_178);  view_174 = view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_173: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_99, [0, 2, 1, 3]);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_239: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_173, primals_6);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_5: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm, [4, 512, 4096]);  mm = None
        view_6: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_5, [4, 512, -1, 128]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_7: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_5: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_8: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_240: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_173, convert_element_type_8);  permute_173 = convert_element_type_8 = None
        sum_67: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 1, 2], True);  mul_240 = None
        view_179: "bf16[128]" = torch.ops.aten.reshape.default(sum_67, [128]);  sum_67 = None
        convert_element_type_232: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_239, torch.float32);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_241: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_232, convert_element_type_7)
        mul_242: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_232, rsqrt_1);  convert_element_type_232 = None
        sum_68: "f32[4, 512, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [3], True);  mul_241 = None
        pow_48: "f32[4, 512, 32, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_243: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Scalar(sum_68, -0.5);  sum_68 = None
        mul_244: "f32[4, 512, 32, 1]" = torch.ops.aten.mul.Tensor(mul_243, pow_48);  mul_243 = pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_37: "f32[4, 512, 32, 128]" = torch.ops.aten.expand.default(mul_244, [4, 512, 32, 128]);  mul_244 = None
        div_45: "f32[4, 512, 32, 128]" = torch.ops.aten.div.Scalar(expand_37, 128);  expand_37 = None
        pow_49: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_7, 1.0);  convert_element_type_7 = None
        mul_245: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
        mul_246: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(div_45, mul_245);  div_45 = mul_245 = None
        add_102: "f32[4, 512, 32, 128]" = torch.ops.aten.add.Tensor(mul_242, mul_246);  mul_242 = mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_233: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_30: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(convert_element_type_233, memory_format = torch.contiguous_format);  convert_element_type_233 = None
        view_180: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_30, [4, 512, 4096]);  clone_30 = None
        view_181: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_180, [2048, 4096]);  view_180 = None
        permute_174: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_181, [1, 0])
        mm_61: "bf16[4096, 2048]" = torch.ops.aten.mm.default(permute_174, view_4);  permute_174 = view_4 = None
        permute_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_176: "bf16[4096, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_62: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_181, permute_176);  view_181 = permute_176 = None
        view_182: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 2048]);  mm_62 = None
        add_103: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_101, view_182);  add_101 = view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_247: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_103, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_3: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_248: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_103, convert_element_type_4);  add_103 = convert_element_type_4 = None
        sum_69: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1], True);  mul_248 = None
        view_183: "bf16[2048]" = torch.ops.aten.reshape.default(sum_69, [2048]);  sum_69 = None
        convert_element_type_238: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_247, torch.float32);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_249: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_238, convert_element_type_3)
        mul_250: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_238, rsqrt);  convert_element_type_238 = None
        sum_70: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
        pow_50: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_251: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_70, -0.5);  sum_70 = None
        mul_252: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_251, pow_50);  mul_251 = pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_38: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_252, [4, 512, 2048]);  mul_252 = None
        div_46: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_38, 2048);  expand_38 = None
        pow_51: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 1.0);  convert_element_type_3 = None
        mul_253: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_51, 2.0);  pow_51 = None
        mul_254: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_46, mul_253);  div_46 = mul_253 = None
        add_104: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_250, mul_254);  mul_250 = mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_239: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None
        add_105: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_95, convert_element_type_239);  add_95 = convert_element_type_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:489 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        convert_element_type_240: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_44: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_24: "f32[4, 512, 2048]" = torch.ops.aten.where.self(unsqueeze_44, full_default_18, convert_element_type_240);  unsqueeze_44 = full_default_18 = convert_element_type_240 = None
        full_default_63: "f32[151936, 2048]" = torch.ops.aten.full.default([151936, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_16: "f32[151936, 2048]" = torch.ops.aten.index_put.default(full_default_63, [primals_1], where_24, True);  full_default_63 = primals_1 = where_24 = None
        convert_element_type_241: "bf16[151936, 2048]" = torch.ops.prims.convert_element_type.default(index_put_16, torch.bfloat16);  index_put_16 = None
        return (None, convert_element_type_241, None, view_183, mm_61, view_179, mm_59, view_175, mm_57, mm_55, view_166, mm_53, _grouped_mm_22, _grouped_mm_20, view_160, mm_51, view_156, mm_49, view_152, mm_47, mm_45, view_143, mm_43, _grouped_mm_18, _grouped_mm_16, view_137, mm_41, view_133, mm_39, view_129, mm_37, mm_35, view_120, mm_33, _grouped_mm_14, _grouped_mm_12, view_114, mm_31, view_110, mm_29, view_106, mm_27, mm_25, view_97, mm_23, _grouped_mm_10, _grouped_mm_8, view_91, mm_21, None)
