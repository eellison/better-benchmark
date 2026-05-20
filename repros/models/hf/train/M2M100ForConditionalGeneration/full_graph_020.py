class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024]", primals_3: "f32[64, 128, 1024]", primals_4: "f32[1024, 1024]", primals_6: "f32[1024, 1024]", primals_8: "f32[1024, 1024]", primals_11: "f32[1024, 1024]", primals_14: "f32[1024]", primals_16: "f32[1024, 1024]", primals_18: "f32[1024, 1024]", primals_20: "f32[1024, 1024]", primals_23: "f32[1024, 1024]", primals_25: "f32[1024]", primals_27: "f32[4096, 1024]", primals_29: "f32[1024, 4096]", getitem_1: "f32[64, 128, 1]", rsqrt: "f32[64, 128, 1]", view: "f32[8192, 1024]", permute_1: "f32[64, 16, 128, 64]", permute_4: "f32[64, 16, 128, 64]", permute_5: "f32[64, 16, 128, 64]", where: "f32[64, 1, 128, 128]", getitem_2: "f32[64, 16, 128, 64]", getitem_3: "f32[64, 16, 128]", getitem_4: "i64[]", getitem_5: "i64[]", addmm_3: "f32[8192, 1024]", gt: "b8[64, 128, 1024]", getitem_7: "f32[64, 128, 1]", rsqrt_1: "f32[64, 128, 1]", view_12: "f32[8192, 1024]", view_15: "f32[8192, 1024]", where_2: "f32[64, 16, 128, 128]", gt_1: "b8[64, 16, 128, 128]", view_28: "f32[8192, 1024]", gt_2: "b8[64, 128, 1024]", mul_12: "f32[64, 128, 1024]", view_30: "f32[8192, 1024]", view_32: "f32[8192, 4096]", gt_3: "b8[64, 128, 1024]", le: "b8[64, 128, 4096]", div_1: "f32[64, 128, 1]", permute_32: "f32[1024, 128, 128]", permute_33: "f32[1024, 64, 128]", permute_34: "f32[1024, 64, 128]", permute_35: "f32[1024, 128, 64]", tangents_1: "f32[64, 128, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:474 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_16: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_17: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_34: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_17, [8192, 1024]);  mul_17 = None
        permute_18: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_19: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_34, permute_19);  permute_19 = None
        permute_20: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_1: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_20, view_32);  permute_20 = view_32 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        view_35: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        view_36: "f32[64, 128, 4096]" = torch.ops.aten.reshape.default(mm, [64, 128, 4096]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        where_3: "f32[64, 128, 4096]" = torch.ops.aten.where.self(le, full_default_1, view_36);  le = full_default_1 = view_36 = None
        view_37: "f32[8192, 4096]" = torch.ops.aten.reshape.default(where_3, [8192, 4096]);  where_3 = None
        permute_17: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_23: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_2: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_37, permute_23);  permute_23 = None
        permute_24: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_24, view_30);  permute_24 = view_30 = None
        sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        view_38: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        view_39: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_2, [64, 128, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_19: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_39, primals_25);  primals_25 = None
        mul_20: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_19, 1024)
        sum_4: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_19, [2], True)
        mul_21: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_19, mul_12);  mul_19 = None
        sum_5: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True);  mul_21 = None
        mul_22: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, sum_5);  sum_5 = None
        sub_5: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_20, sum_4);  mul_20 = sum_4 = None
        sub_6: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_5, mul_22);  sub_5 = mul_22 = None
        mul_23: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_6);  div_1 = sub_6 = None
        mul_24: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_39, mul_12);  mul_12 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_24, [0, 1]);  mul_24 = None
        sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_39, [0, 1]);  view_39 = None
        add_10: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(tangents_1, mul_23);  tangents_1 = mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:465 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_25: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_26: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_10, mul_25);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_40: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_26, [8192, 1024]);  mul_26 = None
        permute_16: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_27: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_4: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_40, permute_27);  permute_27 = None
        permute_28: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_5: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_28, view_28);  permute_28 = view_28 = None
        sum_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41: "f32[1024]" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
        view_42: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [64, 128, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_43: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_42, [64, 128, 16, 64]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "f32[64, 16, 128, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_44: "f32[1024, 128, 64]" = torch.ops.aten.reshape.default(clone_7, [1024, 128, 64]);  clone_7 = None
        bmm_2: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(permute_32, view_44);  permute_32 = None
        bmm_3: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_44, permute_33);  view_44 = permute_33 = None
        view_45: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 128, 64]);  bmm_2 = None
        view_46: "f32[64, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [64, 16, 128, 128]);  bmm_3 = None
        convert_element_type_2: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_27: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_28: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_46, mul_27);  view_46 = mul_27 = None
        mul_29: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_28, where_2);  mul_28 = None
        sum_9: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [-1], True)
        neg: "f32[64, 16, 128, 128]" = torch.ops.aten.neg.default(where_2);  where_2 = None
        fma: "f32[64, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_9, mul_29);  neg = sum_9 = mul_29 = None
        view_47: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(fma, [1024, 128, 128]);  fma = None
        bmm_4: "f32[1024, 64, 128]" = torch.ops.aten.bmm.default(permute_34, view_47);  permute_34 = None
        bmm_5: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(view_47, permute_35);  view_47 = permute_35 = None
        view_48: "f32[64, 16, 64, 128]" = torch.ops.aten.reshape.default(bmm_4, [64, 16, 64, 128]);  bmm_4 = None
        view_49: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 128, 64]);  bmm_5 = None
        mul_30: "f32[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(view_48, 0.3535533905932738);  view_48 = None
        permute_36: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(mul_30, [0, 1, 3, 2]);  mul_30 = None
        mul_31: "f32[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(view_49, 0.3535533905932738);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_37: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_9: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_50: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_9, [64, 128, 1024]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_38: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(permute_36, [0, 2, 1, 3]);  permute_36 = None
        view_51: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_38, [64, 128, 1024]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_52: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_50, [8192, 1024]);  view_50 = None
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_39: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_6: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_52, permute_39);  permute_39 = None
        permute_40: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_40, view_15);  permute_40 = None
        sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        view_53: "f32[1024]" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
        view_54: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_6, [64, 128, 1024]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        clone_10: "f32[64, 128, 1024]" = torch.ops.aten.clone.default(view_51, memory_format = torch.contiguous_format);  view_51 = None
        view_55: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_10, [8192, 1024]);  clone_10 = None
        permute_10: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_43: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_8: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_55, permute_43);  permute_43 = None
        permute_44: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_44, view_15);  permute_44 = view_15 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        view_56: "f32[1024]" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
        view_57: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_8, [64, 128, 1024]);  mm_8 = None
        add_11: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(view_54, view_57);  view_54 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_47: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(mul_31, [0, 2, 1, 3]);  mul_31 = None
        clone_11: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_58: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_11, [64, 128, 1024]);  clone_11 = None
        view_59: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_58, [8192, 1024]);  view_58 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_48: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_10: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_59, permute_48);  permute_48 = None
        permute_49: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_59, [1, 0])
        mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_49, view_12);  permute_49 = view_12 = None
        sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_59, [0], True);  view_59 = None
        view_60: "f32[1024]" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        view_61: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_10, [64, 128, 1024]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        mul_33: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_61, primals_14);  primals_14 = None
        mul_34: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, 1024)
        sum_13: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_3, [64, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:450 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_7);  add_2 = getitem_7 = None
        mul_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_35: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, mul_4);  mul_33 = None
        sum_14: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, sum_14);  sum_14 = None
        sub_8: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_34, sum_13);  mul_34 = sum_13 = None
        sub_9: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        div_2: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_37: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_9);  div_2 = sub_9 = None
        mul_38: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_61, mul_4);  mul_4 = None
        sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_61, [0, 1]);  view_61 = None
        add_12: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_10, mul_37);  add_10 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:450 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_3: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_39: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_40: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_12, mul_39);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_40, [8192, 1024]);  mul_40 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_12: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_62, permute_52);  permute_52 = None
        permute_53: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_62, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_6, [64, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_53, view_10);  permute_53 = view_10 = None
        sum_17: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True);  view_62 = None
        view_63: "f32[1024]" = torch.ops.aten.reshape.default(sum_17, [1024]);  sum_17 = None
        view_64: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, [64, 128, 1024]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_65: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_64, [64, 128, 16, 64]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(where, [64, 16, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_56, permute_1, permute_4, permute_5, expand, getitem_2, getitem_3, getitem_4, getitem_5, 0.1, [True, True, True, False], scale = 0.125);  permute_56 = permute_1 = permute_4 = permute_5 = expand = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_10: "f32[64, 16, 128, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_11: "f32[64, 16, 128, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_12: "f32[64, 16, 128, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_57: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        view_66: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_57, [64, 128, 1024]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_58: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
        view_67: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_58, [64, 128, 1024]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_68: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_66, [8192, 1024]);  view_66 = None
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_59: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_14: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_68, permute_59);  permute_59 = None
        permute_60: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_68, [1, 0])
        mm_15: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_60, view);  permute_60 = None
        sum_18: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True);  view_68 = None
        view_69: "f32[1024]" = torch.ops.aten.reshape.default(sum_18, [1024]);  sum_18 = None
        view_70: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_14, [64, 128, 1024]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_71: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_67, [8192, 1024]);  view_67 = None
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_16: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_71, permute_63);  permute_63 = None
        permute_64: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_71, [1, 0])
        mm_17: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_64, view);  permute_64 = None
        sum_19: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True);  view_71 = None
        view_72: "f32[1024]" = torch.ops.aten.reshape.default(sum_19, [1024]);  sum_19 = None
        view_73: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_16, [64, 128, 1024]);  mm_16 = None
        add_13: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(view_70, view_73);  view_70 = view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_74: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_67, [64, 128, 1024]);  permute_67 = None
        view_75: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_74, [8192, 1024]);  view_74 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_68: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_18: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_75, permute_68);  permute_68 = None
        permute_69: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_19: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_69, view);  permute_69 = view = None
        sum_20: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        view_76: "f32[1024]" = torch.ops.aten.reshape.default(sum_20, [1024]);  sum_20 = None
        view_77: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_18, [64, 128, 1024]);  mm_18 = None
        add_14: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_13, view_77);  add_13 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_42: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_14, primals_1);  primals_1 = None
        mul_43: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_42, 1024)
        sum_21: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_42, [2], True)
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_44: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_42, mul);  mul_42 = None
        sum_22: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_44, [2], True);  mul_44 = None
        mul_45: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_22);  sum_22 = None
        sub_11: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_43, sum_21);  mul_43 = sum_21 = None
        sub_12: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_11, mul_45);  sub_11 = mul_45 = None
        div_3: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_46: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_12);  div_3 = sub_12 = None
        mul_47: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_14, mul);  mul = None
        sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_47, [0, 1]);  mul_47 = None
        sum_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        add_15: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_12, mul_46);  add_12 = mul_46 = None
        return (sum_23, sum_24, add_15, mm_19, view_76, mm_17, view_72, mm_15, view_69, None, mm_13, view_63, add_11, sum_15, sum_16, mm_11, view_60, mm_9, view_56, mm_7, view_53, None, mm_5, view_41, sum_6, sum_7, mm_3, view_38, mm_1, view_35)
