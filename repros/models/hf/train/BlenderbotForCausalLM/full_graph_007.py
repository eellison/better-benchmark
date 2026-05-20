class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2560]", primals_3: "f32[32, 128, 2560]", primals_4: "f32[2560, 2560]", primals_6: "f32[2560, 2560]", primals_8: "f32[2560, 2560]", primals_11: "f32[2560, 2560]", primals_13: "f32[2560]", primals_15: "f32[10240, 2560]", primals_17: "f32[2560, 10240]", getitem_1: "f32[32, 128, 1]", rsqrt: "f32[32, 128, 1]", view: "f32[4096, 2560]", permute_1: "f32[32, 32, 128, 80]", permute_4: "f32[32, 32, 128, 80]", permute_5: "f32[32, 32, 128, 80]", where: "f32[32, 1, 128, 128]", getitem_2: "f32[32, 32, 128, 80]", getitem_3: "f32[32, 32, 128]", getitem_4: "i64[]", getitem_5: "i64[]", addmm_3: "f32[4096, 2560]", gt: "b8[32, 128, 2560]", getitem_7: "f32[32, 128, 1]", rsqrt_1: "f32[32, 128, 1]", view_12: "f32[4096, 2560]", addmm_4: "f32[4096, 10240]", view_14: "f32[4096, 10240]", gt_1: "b8[32, 128, 2560]", tangents_1: "f32[32, 128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:391 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_11: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_12: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(tangents_1, mul_11);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_16: "f32[4096, 2560]" = torch.ops.aten.reshape.default(mul_12, [4096, 2560]);  mul_12 = None
        permute_9: "f32[10240, 2560]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_10: "f32[2560, 10240]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm: "f32[4096, 10240]" = torch.ops.aten.mm.default(view_16, permute_10);  permute_10 = None
        permute_11: "f32[2560, 4096]" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_1: "f32[2560, 10240]" = torch.ops.aten.mm.default(permute_11, view_14);  permute_11 = view_14 = None
        sum_1: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        view_17: "f32[2560]" = torch.ops.aten.reshape.default(sum_1, [2560]);  sum_1 = None
        view_18: "f32[32, 128, 10240]" = torch.ops.aten.reshape.default(mm, [32, 128, 10240]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_13: "f32[32, 128, 10240]" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 10240]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_7: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
        erf: "f32[32, 128, 10240]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_5: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
        mul_15: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(view_13, view_13)
        mul_16: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_15, -0.5);  mul_15 = None
        exp: "f32[32, 128, 10240]" = torch.ops.aten.exp.default(mul_16);  mul_16 = None
        mul_17: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_18: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(view_13, mul_17);  view_13 = mul_17 = None
        add_8: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(mul_14, mul_18);  mul_14 = mul_18 = None
        mul_19: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(view_18, add_8);  view_18 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "f32[4096, 10240]" = torch.ops.aten.reshape.default(mul_19, [4096, 10240]);  mul_19 = None
        permute_8: "f32[2560, 10240]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_14: "f32[10240, 2560]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_2: "f32[4096, 2560]" = torch.ops.aten.mm.default(view_19, permute_14);  permute_14 = None
        permute_15: "f32[10240, 4096]" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_3: "f32[10240, 2560]" = torch.ops.aten.mm.default(permute_15, view_12);  permute_15 = view_12 = None
        sum_2: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        view_20: "f32[10240]" = torch.ops.aten.reshape.default(sum_2, [10240]);  sum_2 = None
        view_21: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(mm_2, [32, 128, 2560]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_21: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(view_21, primals_13);  primals_13 = None
        mul_22: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_21, 2560)
        sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_1: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add_2, getitem_7);  add_2 = getitem_7 = None
        mul_4: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_23: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_21, mul_4);  mul_21 = None
        sum_4: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True);  mul_23 = None
        mul_24: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_4, sum_4);  sum_4 = None
        sub_3: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_22, sum_3);  mul_22 = sum_3 = None
        sub_4: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_3, mul_24);  sub_3 = mul_24 = None
        div: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_25: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_26: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(view_21, mul_4);  mul_4 = None
        sum_5: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_26, [0, 1]);  mul_26 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_21, [0, 1]);  view_21 = None
        add_9: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(tangents_1, mul_25);  tangents_1 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_27: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_28: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(add_9, mul_27);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_22: "f32[4096, 2560]" = torch.ops.aten.reshape.default(mul_28, [4096, 2560]);  mul_28 = None
        permute_7: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_18: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_4: "f32[4096, 2560]" = torch.ops.aten.mm.default(view_22, permute_18);  permute_18 = None
        permute_19: "f32[2560, 4096]" = torch.ops.aten.permute.default(view_22, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[32, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(permute_6, [32, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[4096, 2560]" = torch.ops.aten.reshape.default(view_9, [4096, 2560]);  view_9 = None
        mm_5: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_7: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_23: "f32[2560]" = torch.ops.aten.reshape.default(sum_7, [2560]);  sum_7 = None
        view_24: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(mm_4, [32, 128, 2560]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "f32[32, 128, 32, 80]" = torch.ops.aten.reshape.default(view_24, [32, 128, 32, 80]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_22: "f32[32, 32, 128, 80]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where, [32, 32, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_22, permute_1, permute_4, permute_5, expand, getitem_2, getitem_3, getitem_4, getitem_5, 0.0, [True, True, True, False], scale = 0.11180339887498948);  permute_22 = permute_1 = permute_4 = permute_5 = expand = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_8: "f32[32, 32, 128, 80]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_9: "f32[32, 32, 128, 80]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_10: "f32[32, 32, 128, 80]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_23: "f32[32, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_26: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(permute_23, [32, 128, 2560]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_24: "f32[32, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
        view_27: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(permute_24, [32, 128, 2560]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_28: "f32[4096, 2560]" = torch.ops.aten.reshape.default(view_26, [4096, 2560]);  view_26 = None
        permute_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_25: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_6: "f32[4096, 2560]" = torch.ops.aten.mm.default(view_28, permute_25);  permute_25 = None
        permute_26: "f32[2560, 4096]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_7: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_26, view);  permute_26 = None
        sum_8: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        view_29: "f32[2560]" = torch.ops.aten.reshape.default(sum_8, [2560]);  sum_8 = None
        view_30: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(mm_6, [32, 128, 2560]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_31: "f32[4096, 2560]" = torch.ops.aten.reshape.default(view_27, [4096, 2560]);  view_27 = None
        permute_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_29: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8: "f32[4096, 2560]" = torch.ops.aten.mm.default(view_31, permute_29);  permute_29 = None
        permute_30: "f32[2560, 4096]" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_9: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_30, view);  permute_30 = None
        sum_9: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_31, [0], True);  view_31 = None
        view_32: "f32[2560]" = torch.ops.aten.reshape.default(sum_9, [2560]);  sum_9 = None
        view_33: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(mm_8, [32, 128, 2560]);  mm_8 = None
        add_10: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(view_30, view_33);  view_30 = view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_33: "f32[32, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        view_34: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(permute_33, [32, 128, 2560]);  permute_33 = None
        view_35: "f32[4096, 2560]" = torch.ops.aten.reshape.default(view_34, [4096, 2560]);  view_34 = None
        permute: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_34: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_10: "f32[4096, 2560]" = torch.ops.aten.mm.default(view_35, permute_34);  permute_34 = None
        permute_35: "f32[2560, 4096]" = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_11: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_35, view);  permute_35 = view = None
        sum_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True);  view_35 = None
        view_36: "f32[2560]" = torch.ops.aten.reshape.default(sum_10, [2560]);  sum_10 = None
        view_37: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(mm_10, [32, 128, 2560]);  mm_10 = None
        add_11: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, view_37);  add_10 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_30: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
        mul_31: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_30, 2560)
        sum_11: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True)
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_32: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_30, mul);  mul_30 = None
        sum_12: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_32, [2], True);  mul_32 = None
        mul_33: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, sum_12);  sum_12 = None
        sub_6: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_31, sum_11);  mul_31 = sum_11 = None
        sub_7: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_6, mul_33);  sub_6 = mul_33 = None
        div_1: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_34: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_35: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(add_11, mul);  mul = None
        sum_13: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_35, [0, 1]);  mul_35 = None
        sum_14: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        add_12: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(add_9, mul_34);  add_9 = mul_34 = None
        return (sum_13, sum_14, add_12, mm_11, view_36, mm_9, view_32, mm_7, view_29, None, mm_5, view_23, sum_5, sum_6, mm_3, view_20, mm_1, view_17)
