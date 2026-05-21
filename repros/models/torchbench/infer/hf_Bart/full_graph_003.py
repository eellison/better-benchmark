class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 512, 768]", arg1_1: "f16[768, 768]", arg2_1: "f16[768]", arg3_1: "f16[768, 768]", arg4_1: "f16[768]", arg5_1: "f16[768, 768]", arg6_1: "f16[768]", arg7_1: "b8[1, 1, 512, 512]", arg8_1: "f16[768, 768]", arg9_1: "f16[768]", arg10_1: "f16[768]", arg11_1: "f16[768]", arg12_1: "f16[3072, 768]", arg13_1: "f16[3072]", arg14_1: "f16[768, 3072]", arg15_1: "f16[768]", arg16_1: "f16[768]", arg17_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f16[512, 768]" = torch.ops.aten.reshape.default(arg0_1, [512, 768])
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "f16[512, 768]" = torch.ops.aten.addmm.default(arg2_1, view, permute);  arg2_1 = view = permute = None
        view_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None
        view_2: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, -1, 64]);  view_1 = None
        permute_1: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_9: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        mul: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_9, 0.3535533905932738);  convert_element_type_9 = None
        expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul, [1, 12, 512, 64]);  mul = None
        view_9: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand, [12, 512, 64]);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_3: "f16[512, 768]" = torch.ops.aten.reshape.default(arg0_1, [512, 768])
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "f16[512, 768]" = torch.ops.aten.addmm.default(arg4_1, view_3, permute_2);  arg4_1 = view_3 = permute_2 = None
        view_4: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, 512, -1, 64]);  view_4 = None
        permute_4: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_10: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        permute_6: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_10, [0, 1, 3, 2]);  convert_element_type_10 = None
        mul_1: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_1, [1, 12, 64, 512]);  mul_1 = None
        view_10: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 512]);  expand_1 = None
        bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(arg7_1, full_default_1, full_default);  arg7_1 = full_default_1 = full_default = None
        add: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        eq: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add, -inf)
        logical_not: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        expand_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_1, [1, 12, 512, 512]);  where_1 = None
        view_12: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_2, [12, 512, 512]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_5: "f16[512, 768]" = torch.ops.aten.reshape.default(arg0_1, [512, 768])
        permute_3: "f16[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "f16[512, 768]" = torch.ops.aten.addmm.default(arg6_1, view_5, permute_3);  arg6_1 = view_5 = permute_3 = None
        view_6: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, 512, -1, 64]);  view_6 = None
        permute_5: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_11: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_11, [1, 12, 512, 64]);  convert_element_type_11 = None
        view_13: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 512, 64]);  expand_3 = None
        bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
        convert_element_type_13: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_14, torch.float16);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_13, [0, 2, 1, 3]);  convert_element_type_13 = None
        clone: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone, [1, 512, -1]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_16: "f16[512, 768]" = torch.ops.aten.reshape.default(view_15, [512, 768]);  view_15 = None
        permute_8: "f16[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_3: "f16[512, 768]" = torch.ops.aten.addmm.default(arg9_1, view_16, permute_8);  arg9_1 = view_16 = permute_8 = None
        view_17: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:293 in forward, code: hidden_states = residual + hidden_states
        add_1: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(arg0_1, view_17);  arg0_1 = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:294 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
        sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_1);  convert_element_type_17 = getitem_1 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
        add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        convert_element_type_18: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:297 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_18: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_18, [512, 768])
        permute_9: "f16[768, 3072]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_4: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg13_1, view_18, permute_9);  arg13_1 = view_18 = permute_9 = None
        view_19: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_22: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_23: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:299 in forward, code: hidden_states = self.fc2(hidden_states)
        view_20: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_23, [512, 3072]);  convert_element_type_23 = None
        permute_10: "f16[3072, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_5: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_20, permute_10);  arg15_1 = view_20 = permute_10 = None
        view_21: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:301 in forward, code: hidden_states = residual + hidden_states
        add_5: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_18, view_21);  convert_element_type_18 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:302 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_27: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_27, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_27, getitem_3);  convert_element_type_27 = getitem_3 = None
        add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg16_1);  mul_7 = arg16_1 = None
        add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_8, arg17_1);  mul_8 = arg17_1 = None
        convert_element_type_28: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:304 in forward, code: if hidden_states.dtype == torch.float16 and not torch.isfinite(hidden_states).all():
        eq_1: "b8[1, 512, 768]" = torch.ops.aten.eq.Tensor(convert_element_type_28, convert_element_type_28)
        abs_1: "f16[1, 512, 768]" = torch.ops.aten.abs.default(convert_element_type_28)
        ne: "b8[1, 512, 768]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        mul_9: "b8[1, 512, 768]" = torch.ops.aten.mul.Tensor(eq_1, ne);  eq_1 = ne = None
        logical_not_2: "b8[1, 512, 768]" = torch.ops.aten.logical_not.default(mul_9);  mul_9 = None
        any_2: "b8[]" = torch.ops.aten.any.dims(logical_not_2);  logical_not_2 = None
        logical_not_3: "b8[]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        return (logical_not_3, convert_element_type_28)
