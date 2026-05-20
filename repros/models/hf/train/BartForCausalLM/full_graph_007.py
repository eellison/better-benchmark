class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 1024, 1024]", primals_2: "f32[1024, 1024]", primals_4: "f32[1024, 1024]", primals_6: "f32[1024, 1024]", primals_9: "f32[1024, 1024]", primals_11: "f32[1024]", primals_13: "f32[4096, 1024]", primals_15: "f32[1024, 4096]", primals_17: "f32[1024]", permute_1: "f32[8, 16, 1024, 64]", permute_4: "f32[8, 16, 1024, 64]", permute_5: "f32[8, 16, 1024, 64]", where: "f32[8, 1, 1024, 1024]", getitem: "f32[8, 16, 1024, 64]", getitem_1: "f32[8, 16, 1024]", getitem_2: "i64[]", getitem_3: "i64[]", addmm_3: "f32[8192, 1024]", gt: "b8[8, 1024, 1024]", getitem_5: "f32[8, 1024, 1]", rsqrt: "f32[8, 1024, 1]", view_12: "f32[8192, 1024]", addmm_4: "f32[8192, 4096]", view_14: "f32[8192, 4096]", gt_1: "b8[8, 1024, 1024]", mul_9: "f32[8, 1024, 1024]", div: "f32[8, 1024, 1]", tangents_1: "f32[8, 1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_12: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, primals_17);  primals_17 = None
        mul_13: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_12, 1024)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_12, [2], True)
        mul_14: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_12, mul_9);  mul_12 = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True);  mul_14 = None
        mul_15: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_9, sum_2);  sum_2 = None
        sub_3: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_13, sum_1);  mul_13 = sum_1 = None
        sub_4: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_3, mul_15);  sub_3 = mul_15 = None
        mul_16: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_17: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_9);  mul_9 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1]);  mul_17 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:386 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_18: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_19: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_16, mul_18);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_16: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_19, [8192, 1024]);  mul_19 = None
        permute_9: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_10: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_16, permute_10);  permute_10 = None
        permute_11: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_1: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_11, view_14);  permute_11 = view_14 = None
        sum_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        view_17: "f32[1024]" = torch.ops.aten.reshape.default(sum_5, [1024]);  sum_5 = None
        view_18: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(mm, [8, 1024, 4096]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_13: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_5: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
        erf: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_3: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_21: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_22: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, view_13)
        mul_23: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_22, -0.5);  mul_22 = None
        exp: "f32[8, 1024, 4096]" = torch.ops.aten.exp.default(mul_23);  mul_23 = None
        mul_24: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_25: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, mul_24);  view_13 = mul_24 = None
        add_8: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(mul_21, mul_25);  mul_21 = mul_25 = None
        mul_26: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_18, add_8);  view_18 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_26, [8192, 4096]);  mul_26 = None
        permute_8: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_14: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_2: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_19, permute_14);  permute_14 = None
        permute_15: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_15, view_12);  permute_15 = view_12 = None
        sum_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        view_20: "f32[4096]" = torch.ops.aten.reshape.default(sum_6, [4096]);  sum_6 = None
        view_21: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 1024]);  mm_2 = None
        add_9: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_16, view_21);  mul_16 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_28: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_9, primals_11);  primals_11 = None
        mul_29: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_28, 1024)
        sum_7: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_28, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:362 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(primals_1, mul_1);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add, getitem_5);  add = getitem_5 = None
        mul_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_30: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_28, mul_2);  mul_28 = None
        sum_8: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True);  mul_30 = None
        mul_31: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_2, sum_8);  sum_8 = None
        sub_6: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_29, sum_7);  mul_29 = sum_7 = None
        sub_7: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_6, mul_31);  sub_6 = mul_31 = None
        div_1: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_32: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_33: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_9, mul_2);  mul_2 = None
        sum_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_33, [0, 1]);  mul_33 = None
        sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:362 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_34: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_35: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_32, mul_34);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_22: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_35, [8192, 1024]);  mul_35 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_18: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_4: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_22, permute_18);  permute_18 = None
        permute_19: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_22, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_6, [8, 1024, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        mm_5: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_23: "f32[1024]" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
        view_24: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_4, [8, 1024, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_24, [8, 1024, 16, 64]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_22: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where, [8, 16, 1024, 1024]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_22, permute_1, permute_4, permute_5, expand, getitem, getitem_1, getitem_2, getitem_3, 0.0, [True, True, True, False], scale = 0.125);  permute_22 = permute_1 = permute_4 = permute_5 = expand = getitem = getitem_1 = getitem_2 = getitem_3 = None
        getitem_8: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_9: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_10: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_23: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_26: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_23, [8, 1024, 1024]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_24: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
        view_27: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_24, [8, 1024, 1024]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_28: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_26, [8192, 1024]);  view_26 = None
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_6: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_28, permute_25);  permute_25 = None
        permute_26: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_28, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[8192, 1024]" = torch.ops.aten.reshape.default(primals_1, [8192, 1024]);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_26, view);  permute_26 = None
        sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        view_29: "f32[1024]" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        view_30: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_6, [8, 1024, 1024]);  mm_6 = None
        add_10: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_32, view_30);  mul_32 = view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_31: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_27, [8192, 1024]);  view_27 = None
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_29: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_31, permute_29);  permute_29 = None
        permute_30: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_30, view);  permute_30 = None
        sum_13: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_31, [0], True);  view_31 = None
        view_32: "f32[1024]" = torch.ops.aten.reshape.default(sum_13, [1024]);  sum_13 = None
        view_33: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_8, [8, 1024, 1024]);  mm_8 = None
        add_11: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_10, view_33);  add_10 = view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_33: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        view_34: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_33, [8, 1024, 1024]);  permute_33 = None
        view_35: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_34, [8192, 1024]);  view_34 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_10: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_35, permute_34);  permute_34 = None
        permute_35: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_35, view);  permute_35 = view = None
        sum_14: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True);  view_35 = None
        view_36: "f32[1024]" = torch.ops.aten.reshape.default(sum_14, [1024]);  sum_14 = None
        view_37: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_10, [8, 1024, 1024]);  mm_10 = None
        add_12: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_11, view_37);  add_11 = view_37 = None
        return (add_12, mm_11, view_36, mm_9, view_32, mm_7, view_29, None, mm_5, view_23, sum_9, sum_10, mm_3, view_20, mm_1, view_17, sum_3, sum_4)
