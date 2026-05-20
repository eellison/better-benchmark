class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2560]", primals_3: "f32[16, 128, 2560]", primals_4: "f32[2560, 2560]", primals_6: "f32[2560, 2560]", primals_8: "f32[2560, 2560]", primals_11: "f32[2560, 2560]", primals_13: "f32[2560]", primals_15: "f32[10240, 2560]", primals_17: "f32[2560, 10240]", getitem_1: "f32[16, 128, 1]", rsqrt: "f32[16, 128, 1]", view: "f32[2048, 2560]", where_1: "f32[16, 32, 128, 128]", view_16: "f32[2048, 2560]", addmm_3: "f32[2048, 2560]", gt: "b8[16, 128, 2560]", getitem_3: "f32[16, 128, 1]", rsqrt_1: "f32[16, 128, 1]", view_18: "f32[2048, 2560]", addmm_4: "f32[2048, 10240]", view_20: "f32[2048, 10240]", gt_1: "b8[16, 128, 2560]", permute_25: "f32[512, 80, 128]", permute_26: "f32[512, 80, 128]", permute_27: "f32[512, 128, 80]", tangents_1: "f32[16, 128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:293 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_13: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_14: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(tangents_1, mul_13);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "f32[2048, 2560]" = torch.ops.aten.reshape.default(mul_14, [2048, 2560]);  mul_14 = None
        permute_10: "f32[10240, 2560]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_11: "f32[2560, 10240]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm: "f32[2048, 10240]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        permute_12: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_1: "f32[2560, 10240]" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
        sum_2: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_23: "f32[2560]" = torch.ops.aten.reshape.default(sum_2, [2560]);  sum_2 = None
        view_24: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(mm, [16, 128, 10240]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 10240]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_9: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_6: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_16: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_17: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_18: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_17, -0.5);  mul_17 = None
        exp_1: "f32[16, 128, 10240]" = torch.ops.aten.exp.default(mul_18);  mul_18 = None
        mul_19: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_20: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_19, mul_19);  view_19 = mul_19 = None
        add_9: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(mul_16, mul_20);  mul_16 = mul_20 = None
        mul_21: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_24, add_9);  view_24 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_25: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_21, [2048, 10240]);  mul_21 = None
        permute_9: "f32[2560, 10240]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_15: "f32[10240, 2560]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_2: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
        permute_16: "f32[10240, 2048]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_3: "f32[10240, 2560]" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
        sum_3: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
        view_26: "f32[10240]" = torch.ops.aten.reshape.default(sum_3, [10240]);  sum_3 = None
        view_27: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_2, [16, 128, 2560]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_23: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_27, primals_13);  primals_13 = None
        mul_24: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_23, 2560)
        sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_17: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:285 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_4: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt, view_17);  view_17 = None
        mul_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_3: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(primals_3, mul_5);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_6: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_25: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_23, mul_6);  mul_23 = None
        sum_5: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True);  mul_25 = None
        mul_26: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_6, sum_5);  sum_5 = None
        sub_4: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_24, sum_4);  mul_24 = sum_4 = None
        sub_5: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_4, mul_26);  sub_4 = mul_26 = None
        div_1: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_27: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_28: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_27, mul_6);  mul_6 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_28, [0, 1]);  mul_28 = None
        sum_7: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_27, [0, 1]);  view_27 = None
        add_10: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(tangents_1, mul_27);  tangents_1 = mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:285 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_29: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_30: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_10, mul_29);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "f32[2048, 2560]" = torch.ops.aten.reshape.default(mul_30, [2048, 2560]);  mul_30 = None
        permute_8: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_19: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_4: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
        permute_20: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_5: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
        sum_8: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        view_29: "f32[2560]" = torch.ops.aten.reshape.default(sum_8, [2560]);  sum_8 = None
        view_30: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_4, [16, 128, 2560]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_31: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_30, [16, 128, 32, 80]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_32: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_7, [512, 128, 80]);  clone_7 = None
        expand_2: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, [16, 32, 128, 128])
        view_12: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_2, [512, 128, 128]);  expand_2 = None
        permute_24: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm_2: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
        bmm_3: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
        view_33: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_2, [16, 32, 128, 80]);  bmm_2 = None
        view_34: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [16, 32, 128, 128]);  bmm_3 = None
        mul_31: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(view_34, where_1);  view_34 = None
        sum_9: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_31, [-1], True)
        neg: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg, sum_9, mul_31);  neg = sum_9 = mul_31 = None
        view_35: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(fma, [512, 128, 128]);  fma = None
        bmm_4: "f32[512, 80, 128]" = torch.ops.aten.bmm.default(permute_26, view_35);  permute_26 = None
        bmm_5: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_35, permute_27);  view_35 = permute_27 = None
        view_36: "f32[16, 32, 80, 128]" = torch.ops.aten.reshape.default(bmm_4, [16, 32, 80, 128]);  bmm_4 = None
        view_37: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_5, [16, 32, 128, 80]);  bmm_5 = None
        mul_32: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(view_36, 0.334370152488211);  view_36 = None
        permute_28: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(mul_32, [0, 1, 3, 2]);  mul_32 = None
        mul_33: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(view_37, 0.334370152488211);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_29: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_8: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_38: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_8, [16, 128, 2560]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_30: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_39: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_30, [16, 128, 2560]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_40: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_38, [2048, 2560]);  view_38 = None
        permute_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_31: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_6: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
        permute_32: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_7: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        sum_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41: "f32[2560]" = torch.ops.aten.reshape.default(sum_10, [2560]);  sum_10 = None
        view_42: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_6, [16, 128, 2560]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_9: "f32[16, 128, 2560]" = torch.ops.aten.clone.default(view_39, memory_format = torch.contiguous_format);  view_39 = None
        view_43: "f32[2048, 2560]" = torch.ops.aten.reshape.default(clone_9, [2048, 2560]);  clone_9 = None
        permute_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_35: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_43, permute_35);  permute_35 = None
        permute_36: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_43, [1, 0])
        mm_9: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = None
        sum_11: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
        view_44: "f32[2560]" = torch.ops.aten.reshape.default(sum_11, [2560]);  sum_11 = None
        view_45: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_8, [16, 128, 2560]);  mm_8 = None
        add_11: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(view_42, view_45);  view_42 = view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_39: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul_33, [0, 2, 1, 3]);  mul_33 = None
        clone_10: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_46: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_10, [16, 128, 2560]);  clone_10 = None
        view_47: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_46, [2048, 2560]);  view_46 = None
        permute: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_40: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_10: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
        permute_41: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_11: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_12: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
        view_48: "f32[2560]" = torch.ops.aten.reshape.default(sum_12, [2560]);  sum_12 = None
        view_49: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_10, [16, 128, 2560]);  mm_10 = None
        add_12: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_11, view_49);  add_11 = view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_35: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_12, primals_1);  primals_1 = None
        mul_36: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_35, 2560)
        sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True)
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_37: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_35, mul);  mul_35 = None
        sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_37, [2], True);  mul_37 = None
        mul_38: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
        sub_7: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_36, sum_13);  mul_36 = sum_13 = None
        sub_8: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_7, mul_38);  sub_7 = mul_38 = None
        div_2: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_39: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_40: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_12, mul);  mul = None
        sum_15: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_40, [0, 1]);  mul_40 = None
        sum_16: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1]);  add_12 = None
        add_13: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, mul_39);  add_10 = mul_39 = None
        return (sum_15, sum_16, add_13, mm_11, view_48, mm_9, view_44, mm_7, view_41, None, mm_5, view_29, sum_6, sum_7, mm_3, view_26, mm_1, view_23)
