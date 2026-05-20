class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024]", primals_3: "f32[64, 128, 1024]", primals_4: "f32[1024, 1024]", primals_6: "f32[1024, 1024]", primals_8: "f32[1024, 1024]", primals_10: "b8[64, 1, 128, 128]", primals_11: "f32[1024, 1024]", primals_13: "f32[1024]", primals_15: "f32[4096, 1024]", primals_17: "f32[1024, 4096]", getitem_1: "f32[64, 128, 1]", rsqrt: "f32[64, 128, 1]", view: "f32[8192, 1024]", bmm: "f32[1024, 128, 128]", amax: "f32[64, 16, 128, 1]", sum_1: "f32[64, 16, 128, 1]", logical_not_1: "b8[64, 16, 128, 1]", gt: "b8[64, 16, 128, 128]", view_16: "f32[8192, 1024]", addmm_3: "f32[8192, 1024]", gt_1: "b8[64, 128, 1024]", getitem_3: "f32[64, 128, 1]", rsqrt_1: "f32[64, 128, 1]", view_18: "f32[8192, 1024]", view_20: "f32[8192, 4096]", gt_2: "b8[64, 128, 1024]", le: "b8[64, 128, 4096]", permute_24: "f32[1024, 128, 128]", permute_25: "f32[1024, 64, 128]", permute_26: "f32[1024, 64, 128]", permute_27: "f32[1024, 128, 64]", tangents_1: "f32[64, 128, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:376 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_12: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_13: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_12);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_13, [8192, 1024]);  mul_13 = None
        permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_11: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        permute_12: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_1: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_23: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        view_24: "f32[64, 128, 4096]" = torch.ops.aten.reshape.default(mm, [64, 128, 4096]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        where_2: "f32[64, 128, 4096]" = torch.ops.aten.where.self(le, full_default_1, view_24);  le = view_24 = None
        view_25: "f32[8192, 4096]" = torch.ops.aten.reshape.default(where_2, [8192, 4096]);  where_2 = None
        permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_15: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_2: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
        permute_16: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
        sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
        view_26: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        view_27: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_2, [64, 128, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_15: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_27, primals_13);  primals_13 = None
        mul_16: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, 1024)
        sum_4: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_17: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_3, [64, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:368 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_6: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_1, view_17);  view_17 = None
        mul_7: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_3: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(primals_3, mul_7);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_2: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_8: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_17: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, mul_8);  mul_15 = None
        sum_5: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_8, sum_5);  sum_5 = None
        sub_4: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_16, sum_4);  mul_16 = sum_4 = None
        sub_5: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_18);  sub_4 = mul_18 = None
        div_1: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_19: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_20: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_27, mul_8);  mul_8 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_27, [0, 1]);  view_27 = None
        add_7: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(tangents_1, mul_19);  tangents_1 = mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:368 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_21: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_22: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_7, mul_21);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_22, [8192, 1024]);  mul_22 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_4: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
        permute_20: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_5: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
        sum_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        view_29: "f32[1024]" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
        view_30: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [64, 128, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_31: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_30, [64, 128, 16, 64]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "f32[64, 16, 128, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_32: "f32[1024, 128, 64]" = torch.ops.aten.reshape.default(clone_7, [1024, 128, 64]);  clone_7 = None
        bmm_2: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
        bmm_3: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
        view_33: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 128, 64]);  bmm_2 = None
        view_34: "f32[64, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [64, 16, 128, 128]);  bmm_3 = None
        convert_element_type_2: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_23: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_24: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_34, mul_23);  view_34 = mul_23 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 1, 128, 128]" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = full_default_1 = full_default = None
        view_11: "f32[64, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [64, 16, 128, 128]);  bmm = None
        add_2: "f32[64, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        sub_1: "f32[64, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_2, amax);  add_2 = amax = None
        exp: "f32[64, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[64, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_2: "f32[64, 16, 128, 128]" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[64, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        mul_25: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_24, where_1);  mul_24 = None
        sum_9: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [-1], True)
        neg: "f32[64, 16, 128, 128]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma: "f32[64, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_9, mul_25);  neg = sum_9 = mul_25 = None
        view_35: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(fma, [1024, 128, 128]);  fma = None
        bmm_4: "f32[1024, 64, 128]" = torch.ops.aten.bmm.default(permute_26, view_35);  permute_26 = None
        bmm_5: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(view_35, permute_27);  view_35 = permute_27 = None
        view_36: "f32[64, 16, 64, 128]" = torch.ops.aten.reshape.default(bmm_4, [64, 16, 64, 128]);  bmm_4 = None
        view_37: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 128, 64]);  bmm_5 = None
        mul_26: "f32[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(view_36, 0.3535533905932738);  view_36 = None
        permute_28: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(mul_26, [0, 1, 3, 2]);  mul_26 = None
        mul_27: "f32[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(view_37, 0.3535533905932738);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_29: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_9: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_38: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_9, [64, 128, 1024]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_30: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_39: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_30, [64, 128, 1024]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_40: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_38, [8192, 1024]);  view_38 = None
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_31: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_6: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
        permute_32: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41: "f32[1024]" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
        view_42: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_6, [64, 128, 1024]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        clone_10: "f32[64, 128, 1024]" = torch.ops.aten.clone.default(view_39, memory_format = torch.contiguous_format);  view_39 = None
        view_43: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_10, [8192, 1024]);  clone_10 = None
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_35: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_43, permute_35);  permute_35 = None
        permute_36: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_43, [1, 0])
        mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
        view_44: "f32[1024]" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
        view_45: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_8, [64, 128, 1024]);  mm_8 = None
        add_8: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(view_42, view_45);  view_42 = view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_39: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(mul_27, [0, 2, 1, 3]);  mul_27 = None
        clone_11: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_46: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_11, [64, 128, 1024]);  clone_11 = None
        view_47: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_46, [8192, 1024]);  view_46 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_40: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_10: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
        permute_41: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
        view_48: "f32[1024]" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        view_49: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_10, [64, 128, 1024]);  mm_10 = None
        add_9: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_8, view_49);  add_8 = view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_29: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_9, primals_1);  primals_1 = None
        mul_30: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_29, 1024)
        sum_13: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True)
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_31: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_29, mul);  mul_29 = None
        sum_14: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_31, [2], True);  mul_31 = None
        mul_32: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
        sub_7: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_30, sum_13);  mul_30 = sum_13 = None
        sub_8: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_7, mul_32);  sub_7 = mul_32 = None
        div_2: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_33: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_34: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_9, mul);  mul = None
        sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_34, [0, 1]);  mul_34 = None
        sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        add_10: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_7, mul_33);  add_7 = mul_33 = None
        return (sum_15, sum_16, add_10, mm_11, view_48, mm_9, view_44, mm_7, view_41, None, mm_5, view_29, sum_6, sum_7, mm_3, view_26, mm_1, view_23)
