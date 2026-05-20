class GraphModule(torch.nn.Module):
    def forward(self, addmm_56: "f32[128, 2]", primals_315: "i64[1]", primals_316: "i64[1]", full_default_1: "f32[]", rsqrt_28: "f32[1, 128, 1]", rsqrt_27: "f32[1, 128, 1]", rsqrt_26: "f32[1, 128, 1]", rsqrt_25: "f32[1, 128, 1]", rsqrt_24: "f32[1, 128, 1]", rsqrt_23: "f32[1, 128, 1]", rsqrt_22: "f32[1, 128, 1]", rsqrt_21: "f32[1, 128, 1]", rsqrt_20: "f32[1, 128, 1]", rsqrt_19: "f32[1, 128, 1]", rsqrt_18: "f32[1, 128, 1]", rsqrt_17: "f32[1, 128, 1]", rsqrt_16: "f32[1, 128, 1]", rsqrt_15: "f32[1, 128, 1]", rsqrt_14: "f32[1, 128, 1]", rsqrt_13: "f32[1, 128, 1]", rsqrt_12: "f32[1, 128, 1]", rsqrt_11: "f32[1, 128, 1]", rsqrt_10: "f32[1, 128, 1]", rsqrt_9: "f32[1, 128, 1]", rsqrt_8: "f32[1, 128, 1]", rsqrt_7: "f32[1, 128, 1]", rsqrt_6: "f32[1, 128, 1]", rsqrt_5: "f32[1, 128, 1]", rsqrt_4: "f32[1, 128, 1]", rsqrt_3: "f32[1, 128, 1]", rsqrt_2: "f32[1, 128, 1]", rsqrt_1: "f32[1, 128, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        reshape_default: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(addmm_56, _shape_param_0);  addmm_56 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:816 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 1, -1);  reshape_default = None
        getitem: "f32[1, 128, 1]" = split_tensor[0]
        getitem_1: "f32[1, 128, 1]" = split_tensor[1];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:817 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        squeeze_dim: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem, -1);  getitem = None
        clone_default: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:818 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        squeeze_dim_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_1, -1);  getitem_1 = None
        clone_default_1: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:829 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min_default: "i64[1]" = torch.ops.aten.clamp_min.default(primals_315, 0);  primals_315 = None
        clamp_max_default: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 128);  clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:830 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_default_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_316, 0);  primals_316 = None
        clamp_max_default_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default_1, 128);  clamp_min_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        amax_default: "f32[1, 1]" = torch.ops.aten.amax.default(clone_default, [1], True)
        sub_tensor: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[1, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_default, 128)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[1]" = torch.ops.aten.where.self(ne_scalar, clamp_max_default, full_default);  clamp_max_default = None
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[1, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[1]" = torch.ops.aten.neg.default(squeeze_dim_2);  squeeze_dim_2 = None
        where_self_1: "f32[1]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        amax_default_1: "f32[1, 1]" = torch.ops.aten.amax.default(clone_default_1, [1], True)
        sub_tensor_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_default_1, amax_default_1);  clone_default_1 = amax_default_1 = None
        exp_default_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [1], True);  exp_default_1 = None
        log_default_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default_1);  sub_tensor_2 = log_default_1 = None
        ne_scalar_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_default_1, 128)
        where_self_2: "i64[1]" = torch.ops.aten.where.self(ne_scalar_1, clamp_max_default_1, full_default);  clamp_max_default_1 = full_default = None
        unsqueeze_default_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_self_2, 1);  where_self_2 = None
        gather_default_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_tensor_3, 1, unsqueeze_default_1);  sub_tensor_3 = unsqueeze_default_1 = None
        squeeze_dim_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_default_1, 1);  gather_default_1 = None
        neg_default_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_dim_3);  squeeze_dim_3 = None
        where_self_3: "f32[1]" = torch.ops.aten.where.self(ne_scalar_1, neg_default_1, full_default_1);  neg_default_1 = full_default_1 = None
        sum_default_2: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_2, torch.float32);  sum_default_2 = None
        sum_default_3: "f32[]" = torch.ops.aten.sum.default(where_self_3);  where_self_3 = None
        div_tensor_1: "f32[]" = torch.ops.aten.div.Tensor(sum_default_3, convert_element_type_default_1);  sum_default_3 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:835 in forward, code: total_loss = (start_loss + end_loss) / 2
        add_tensor: "f32[]" = torch.ops.aten.add.Tensor(div_tensor, div_tensor_1);  div_tensor = div_tensor_1 = None
        div_tensor_2: "f32[]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_tensor_3: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 4096);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_4: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 4096);  rsqrt_27 = None
        div_tensor_5: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 4096);  rsqrt_26 = None
        div_tensor_6: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 4096);  rsqrt_25 = None
        div_tensor_7: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None
        div_tensor_8: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None
        div_tensor_9: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None
        div_tensor_10: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None
        div_tensor_11: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None
        div_tensor_12: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None
        div_tensor_13: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None
        div_tensor_14: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None
        div_tensor_15: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None
        div_tensor_16: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None
        div_tensor_17: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None
        div_tensor_18: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None
        div_tensor_19: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None
        div_tensor_20: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None
        div_tensor_21: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None
        div_tensor_22: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None
        div_tensor_23: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None
        div_tensor_24: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None
        div_tensor_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None
        div_tensor_26: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None
        div_tensor_27: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None
        div_tensor_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None
        div_tensor_29: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None
        div_tensor_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None
        return (div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22, div_tensor_23, div_tensor_24, div_tensor_25, div_tensor_26, div_tensor_27, div_tensor_28, div_tensor_29, div_tensor_30)
