class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[768][1]cuda:0", primals_3: "f32[4, 2048, 768][1572864, 768, 1]cuda:0", primals_13: "f32[768][1]cuda:0", getitem_1: "f32[4, 2048, 1][2048, 1, 1]cuda:0", rsqrt: "f32[4, 2048, 1][2048, 1, 1]cuda:0", view: "bf16[8192, 768][768, 1]cuda:0", permute_1: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0", permute_4: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0", permute_5: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0", where: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", getitem_2: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0", getitem_3: "f32[4, 12, 2048, 1][24576, 2048, 1, 1]cuda:0", getitem_8: "i64[][]cuda:0", getitem_9: "i64[][]cuda:0", addmm_3: "bf16[8192, 768][768, 1]cuda:0", gt: "b8[4, 2048, 768][1572864, 768, 1]cuda:0", getitem_12: "f32[8192, 1][1, 1]cuda:0", rsqrt_1: "f32[8192, 1][1, 1]cuda:0", convert_element_type_25: "bf16[8192, 768][768, 1]cuda:0", relu: "bf16[8192, 3072][3072, 1]cuda:0", gt_1: "b8[8192, 768][768, 1]cuda:0", permute_10: "bf16[768, 3072][3072, 1]cuda:0", permute_14: "bf16[3072, 768][768, 1]cuda:0", permute_18: "bf16[768, 768][768, 1]cuda:0", permute_25: "bf16[768, 768][768, 1]cuda:0", permute_29: "bf16[768, 768][768, 1]cuda:0", permute_34: "bf16[768, 768][768, 1]cuda:0", tangents_1: "f32[4, 2048, 768][1572864, 768, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        view_14: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(tangents_1, [8192, 768]);  tangents_1 = None
        convert_element_type_34: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:245 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_35: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_10: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_9);  convert_element_type_34 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        mm: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(mul_10, permute_10);  permute_10 = None
        permute_11: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(mul_10, [1, 0])
        mm_1: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_11, relu);  permute_11 = None
        sum_1: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_10, [0], True, dtype = torch.float32);  mul_10 = None
        view_15: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [768]);  sum_1 = None
        convert_element_type_40: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_15, torch.bfloat16);  view_15 = None
        convert_element_type_41: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_42: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_40, torch.float32);  convert_element_type_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        le: "b8[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        where_1: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.where.self(le, full_default_1, mm);  le = full_default_1 = mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        mm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(where_1, permute_14);  permute_14 = None
        permute_15: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_3: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_15, convert_element_type_25);  permute_15 = convert_element_type_25 = None
        sum_2: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0], True, dtype = torch.float32);  where_1 = None
        view_16: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [3072]);  sum_2 = None
        convert_element_type_47: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        convert_element_type_48: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_49: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_50: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_12: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_48, primals_13);  primals_13 = None
        mul_13: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, 768)
        sum_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_12, [1], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [4, 2048, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_3: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_4: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_4);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_12: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_1: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_12, getitem_12);  view_12 = getitem_12 = None
        mul_5: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_14: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, mul_5);  mul_12 = None
        sum_4: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_14, [1], True);  mul_14 = None
        mul_15: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, sum_4);  sum_4 = None
        sub_3: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_13, sum_3);  mul_13 = sum_3 = None
        sub_4: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_3, mul_15);  sub_3 = mul_15 = None
        div: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_16: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_17: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_48, mul_5);  mul_5 = None
        sum_5: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_17, [0]);  mul_17 = None
        sum_6: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_48, [0]);  convert_element_type_48 = None
        add_6: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_14, mul_16);  view_14 = mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_17: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_6, [4, 2048, 768]);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_51: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_17, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_52: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_18: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_19: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, mul_18);  convert_element_type_51 = mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_18: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [8192, 768]);  mul_19 = None
        mm_4: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_18, permute_18);  permute_18 = None
        permute_19: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_18, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_9: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [4, 2048, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [8192, 768]);  view_9 = None
        mm_5: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_7: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
        convert_element_type_57: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [4, 2048, 768]);  mm_4 = None
        convert_element_type_58: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_59: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_57, torch.float32);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_21: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_20, [4, 2048, 12, 64]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_22: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_22, permute_1, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, where, None, None, 2048, 2048, 0.0, False, scale = 1.0);  permute_22 = permute_1 = permute_4 = permute_5 = getitem_2 = getitem_3 = getitem_8 = getitem_9 = where = None
        getitem_13: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_14: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_15: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_23: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        view_22: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_23, [4, 2048, 768]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_24: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3]);  getitem_14 = None
        view_23: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_24, [4, 2048, 768]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_24: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [8192, 768]);  view_22 = None
        mm_6: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_25);  permute_25 = None
        permute_26: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_24, [1, 0])
        mm_7: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_26, view);  permute_26 = None
        sum_8: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_24, [0], True, dtype = torch.float32);  view_24 = None
        view_25: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [768]);  sum_8 = None
        convert_element_type_64: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        view_26: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [4, 2048, 768]);  mm_6 = None
        convert_element_type_65: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None
        convert_element_type_66: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_67: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_64, torch.float32);  convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_27: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [8192, 768]);  view_23 = None
        mm_8: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_27, permute_29);  permute_29 = None
        permute_30: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_9: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_30, view);  permute_30 = None
        sum_9: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_27, [0], True, dtype = torch.float32);  view_27 = None
        view_28: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_9, [768]);  sum_9 = None
        convert_element_type_72: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        view_29: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [4, 2048, 768]);  mm_8 = None
        convert_element_type_73: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        add_7: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_65, convert_element_type_73);  convert_element_type_65 = convert_element_type_73 = None
        convert_element_type_74: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_75: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_72, torch.float32);  convert_element_type_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_33: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
        view_30: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [4, 2048, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        mul_20: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_30, 0.125);  view_30 = None
        view_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_20, [8192, 768]);  mul_20 = None
        mm_10: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_31, permute_34);  permute_34 = None
        permute_35: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_11: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_35, view);  permute_35 = view = None
        sum_10: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_31, [0], True, dtype = torch.float32);  view_31 = None
        view_32: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
        convert_element_type_80: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_32, torch.bfloat16);  view_32 = None
        view_33: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [4, 2048, 768]);  mm_10 = None
        convert_element_type_81: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        add_8: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, convert_element_type_81);  add_7 = convert_element_type_81 = None
        convert_element_type_82: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_83: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_80, torch.float32);  convert_element_type_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_22: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, primals_1);  primals_1 = None
        mul_23: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, 768)
        sum_11: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True)
        sub: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_24: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, mul);  mul_22 = None
        sum_12: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        mul_25: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_12);  sum_12 = None
        sub_6: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_23, sum_11);  mul_23 = sum_11 = None
        sub_7: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_6, mul_25);  sub_6 = mul_25 = None
        div_1: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_26: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_27: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, mul);  mul = None
        sum_13: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_27, [0, 1]);  mul_27 = None
        sum_14: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1]);  add_8 = None
        add_9: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, mul_26);  view_17 = mul_26 = None
        return (sum_13, sum_14, add_9, convert_element_type_82, convert_element_type_83, convert_element_type_74, convert_element_type_75, convert_element_type_66, convert_element_type_67, None, convert_element_type_58, convert_element_type_59, sum_5, sum_6, convert_element_type_49, convert_element_type_50, convert_element_type_41, convert_element_type_42)
