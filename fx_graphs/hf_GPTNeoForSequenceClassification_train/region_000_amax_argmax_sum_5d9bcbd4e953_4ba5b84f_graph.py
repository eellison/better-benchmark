class GraphModule(torch.nn.Module):
    def forward(self, mm_72: "f32[4096, 2]", primals_1: "i64[32, 128]", iota_1: "i64[32]", primals_343: "i64[32]", full_default_1: "f32[]", rsqrt_48: "f32[32, 128, 1]", rsqrt_47: "f32[32, 128, 1]", view_519: "f32[512, 128, 128]", view_515: "f32[512, 128, 128]", view_516: "f32[512, 128, 128]", rsqrt_46: "f32[32, 128, 1]", rsqrt_45: "f32[32, 128, 1]", view_497: "f32[512, 128, 128]", view_493: "f32[512, 128, 128]", view_494: "f32[512, 128, 128]", rsqrt_44: "f32[32, 128, 1]", rsqrt_43: "f32[32, 128, 1]", view_475: "f32[512, 128, 128]", view_471: "f32[512, 128, 128]", view_472: "f32[512, 128, 128]", rsqrt_42: "f32[32, 128, 1]", rsqrt_41: "f32[32, 128, 1]", view_453: "f32[512, 128, 128]", view_449: "f32[512, 128, 128]", view_450: "f32[512, 128, 128]", rsqrt_40: "f32[32, 128, 1]", rsqrt_39: "f32[32, 128, 1]", view_431: "f32[512, 128, 128]", view_427: "f32[512, 128, 128]", view_428: "f32[512, 128, 128]", rsqrt_38: "f32[32, 128, 1]", rsqrt_37: "f32[32, 128, 1]", view_409: "f32[512, 128, 128]", view_405: "f32[512, 128, 128]", view_406: "f32[512, 128, 128]", rsqrt_36: "f32[32, 128, 1]", rsqrt_35: "f32[32, 128, 1]", view_387: "f32[512, 128, 128]", view_383: "f32[512, 128, 128]", view_384: "f32[512, 128, 128]", rsqrt_34: "f32[32, 128, 1]", rsqrt_33: "f32[32, 128, 1]", view_365: "f32[512, 128, 128]", view_361: "f32[512, 128, 128]", view_362: "f32[512, 128, 128]", rsqrt_32: "f32[32, 128, 1]", rsqrt_31: "f32[32, 128, 1]", view_343: "f32[512, 128, 128]", view_339: "f32[512, 128, 128]", view_340: "f32[512, 128, 128]", rsqrt_30: "f32[32, 128, 1]", rsqrt_29: "f32[32, 128, 1]", view_321: "f32[512, 128, 128]", view_317: "f32[512, 128, 128]", view_318: "f32[512, 128, 128]", rsqrt_28: "f32[32, 128, 1]", rsqrt_27: "f32[32, 128, 1]", view_299: "f32[512, 128, 128]", view_295: "f32[512, 128, 128]", view_296: "f32[512, 128, 128]", rsqrt_26: "f32[32, 128, 1]", rsqrt_25: "f32[32, 128, 1]", view_277: "f32[512, 128, 128]", view_273: "f32[512, 128, 128]", view_274: "f32[512, 128, 128]", rsqrt_24: "f32[32, 128, 1]", rsqrt_23: "f32[32, 128, 1]", view_255: "f32[512, 128, 128]", view_251: "f32[512, 128, 128]", view_252: "f32[512, 128, 128]", rsqrt_22: "f32[32, 128, 1]", rsqrt_21: "f32[32, 128, 1]", view_233: "f32[512, 128, 128]", view_229: "f32[512, 128, 128]", view_230: "f32[512, 128, 128]", rsqrt_20: "f32[32, 128, 1]", rsqrt_19: "f32[32, 128, 1]", view_211: "f32[512, 128, 128]", view_207: "f32[512, 128, 128]", view_208: "f32[512, 128, 128]", rsqrt_18: "f32[32, 128, 1]", rsqrt_17: "f32[32, 128, 1]", view_189: "f32[512, 128, 128]", view_185: "f32[512, 128, 128]", view_186: "f32[512, 128, 128]", rsqrt_16: "f32[32, 128, 1]", rsqrt_15: "f32[32, 128, 1]", view_167: "f32[512, 128, 128]", view_163: "f32[512, 128, 128]", view_164: "f32[512, 128, 128]", rsqrt_14: "f32[32, 128, 1]", rsqrt_13: "f32[32, 128, 1]", view_145: "f32[512, 128, 128]", view_141: "f32[512, 128, 128]", view_142: "f32[512, 128, 128]", rsqrt_12: "f32[32, 128, 1]", rsqrt_11: "f32[32, 128, 1]", view_123: "f32[512, 128, 128]", view_119: "f32[512, 128, 128]", view_120: "f32[512, 128, 128]", rsqrt_10: "f32[32, 128, 1]", rsqrt_9: "f32[32, 128, 1]", view_101: "f32[512, 128, 128]", view_97: "f32[512, 128, 128]", view_98: "f32[512, 128, 128]", rsqrt_8: "f32[32, 128, 1]", rsqrt_7: "f32[32, 128, 1]", view_79: "f32[512, 128, 128]", view_75: "f32[512, 128, 128]", view_76: "f32[512, 128, 128]", rsqrt_6: "f32[32, 128, 1]", rsqrt_5: "f32[32, 128, 1]", view_57: "f32[512, 128, 128]", view_53: "f32[512, 128, 128]", view_54: "f32[512, 128, 128]", rsqrt_4: "f32[32, 128, 1]", rsqrt_3: "f32[32, 128, 1]", view_35: "f32[512, 128, 128]", view_31: "f32[512, 128, 128]", view_32: "f32[512, 128, 128]", rsqrt_2: "f32[32, 128, 1]", rsqrt_1: "f32[32, 128, 1]", view_13: "f32[512, 128, 128]", view_9: "f32[512, 128, 128]", view_10: "f32[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:675 in forward, code: logits = self.score(hidden_states)
        reshape_default: "f32[32, 128, 2]" = torch.ops.aten.reshape.default(mm_72, _shape_param_0);  mm_72 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:688 in forward, code: non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        ne_scalar: "b8[32, 128]" = torch.ops.aten.ne.Scalar(primals_1, 0);  primals_1 = None
        convert_element_type_default: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:689 in forward, code: token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        iota_default: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:690 in forward, code: last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        mul_tensor: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota_default, convert_element_type_default);  iota_default = convert_element_type_default = None
        argmax_default: "i64[32]" = torch.ops.aten.argmax.default(mul_tensor, -1);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:698 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        index_tensor: "f32[32, 2]" = torch.ops.aten.index.Tensor(reshape_default, [iota_1, argmax_default]);  reshape_default = iota_1 = argmax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        amax_default: "f32[32, 1]" = torch.ops.aten.amax.default(index_tensor, [1], True)
        sub_tensor: "f32[32, 2]" = torch.ops.aten.sub.Tensor(index_tensor, amax_default);  index_tensor = amax_default = None
        exp_default: "f32[32, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[32]" = torch.ops.aten.ne.Scalar(primals_343, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32]" = torch.ops.aten.where.self(ne_scalar_1, primals_343, full_default)
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[32]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[32]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[32]" = torch.ops.aten.where.self(ne_scalar_1, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_1);  sum_default_1 = convert_element_type_default_1 = None
        unsqueeze_default_1: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(primals_343, 1);  primals_343 = None
        ne_scalar_2: "b8[32, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default_1, -100)
        where_self_2: "i64[32, 1]" = torch.ops.aten.where.self(ne_scalar_2, unsqueeze_default_1, full_default);  ne_scalar_2 = unsqueeze_default_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 2]" = torch.ops.aten.reshape.default(iota_default_1, _shape_param_1);  iota_default_1 = _shape_param_1 = None
        expand_default: "i64[32, 2]" = torch.ops.aten.expand.default(where_self_2, _shape_param_2);  where_self_2 = _shape_param_2 = None
        eq_tensor: "b8[32, 2]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 2048);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_2: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_1: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_default_2: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_3: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 2048);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_4: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 2048);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_3: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_4: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
        permute_default_5: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_5: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 2048);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_6: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 2048);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_6: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_7: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        permute_default_8: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_7: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 2048);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_8: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 2048);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_9: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_10: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        permute_default_11: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_9: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 2048);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_10: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 2048);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_12: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_13: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
        permute_default_14: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_11: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 2048);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_12: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 2048);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_15: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_16: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        permute_default_17: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_13: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 2048);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_14: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 2048);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_18: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_19: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_default_20: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_15: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 2048);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_16: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 2048);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_21: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_22: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
        permute_default_23: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_17: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 2048);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_18: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 2048);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_24: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_25: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_default_26: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_19: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 2048);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_20: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 2048);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_27: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_28: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        permute_default_29: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_21: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 2048);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_22: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 2048);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_30: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_31: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_default_32: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_23: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 2048);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_24: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 2048);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_33: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_34: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        permute_default_35: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_25: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 2048);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_26: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 2048);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_36: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_37: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_default_38: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_27: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 2048);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_28: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 2048);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_39: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_40: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_default_41: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_29: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 2048);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_30: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 2048);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_42: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_43: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_default_44: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_31: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 2048);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_32: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 2048);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_45: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_46: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_default_47: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_33: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 2048);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_34: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 2048);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_48: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_49: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_default_50: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_35: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 2048);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_36: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 2048);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_51: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_52: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_default_53: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_37: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 2048);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_38: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 2048);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_54: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_55: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_default_56: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_39: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 2048);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_40: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 2048);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_57: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_58: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_59: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_41: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 2048);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_42: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 2048);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_60: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_61: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_default_62: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_43: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 2048);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_44: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 2048);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_63: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_64: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_default_65: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_45: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 2048);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_46: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 2048);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_66: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_67: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_68: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_47: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2048);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_48: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_69: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_70: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_71: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (div_tensor, eq_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, div_tensor_3, div_tensor_4, permute_default_3, permute_default_4, permute_default_5, div_tensor_5, div_tensor_6, permute_default_6, permute_default_7, permute_default_8, div_tensor_7, div_tensor_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_9, div_tensor_10, permute_default_12, permute_default_13, permute_default_14, div_tensor_11, div_tensor_12, permute_default_15, permute_default_16, permute_default_17, div_tensor_13, div_tensor_14, permute_default_18, permute_default_19, permute_default_20, div_tensor_15, div_tensor_16, permute_default_21, permute_default_22, permute_default_23, div_tensor_17, div_tensor_18, permute_default_24, permute_default_25, permute_default_26, div_tensor_19, div_tensor_20, permute_default_27, permute_default_28, permute_default_29, div_tensor_21, div_tensor_22, permute_default_30, permute_default_31, permute_default_32, div_tensor_23, div_tensor_24, permute_default_33, permute_default_34, permute_default_35, div_tensor_25, div_tensor_26, permute_default_36, permute_default_37, permute_default_38, div_tensor_27, div_tensor_28, permute_default_39, permute_default_40, permute_default_41, div_tensor_29, div_tensor_30, permute_default_42, permute_default_43, permute_default_44, div_tensor_31, div_tensor_32, permute_default_45, permute_default_46, permute_default_47, div_tensor_33, div_tensor_34, permute_default_48, permute_default_49, permute_default_50, div_tensor_35, div_tensor_36, permute_default_51, permute_default_52, permute_default_53, div_tensor_37, div_tensor_38, permute_default_54, permute_default_55, permute_default_56, div_tensor_39, div_tensor_40, permute_default_57, permute_default_58, permute_default_59, div_tensor_41, div_tensor_42, permute_default_60, permute_default_61, permute_default_62, div_tensor_43, div_tensor_44, permute_default_63, permute_default_64, permute_default_65, div_tensor_45, div_tensor_46, permute_default_66, permute_default_67, permute_default_68, div_tensor_47, div_tensor_48, permute_default_69, permute_default_70, permute_default_71)
