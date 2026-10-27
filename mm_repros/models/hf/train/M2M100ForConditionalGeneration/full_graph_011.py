class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024][1]cuda:0", primals_3: "f32[64, 128, 1024][131072, 1024, 1]cuda:0", primals_13: "f32[1024][1]cuda:0", getitem_1: "f32[64, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[64, 128, 1][128, 1, 1]cuda:0", view: "bf16[8192, 1024][1024, 1]cuda:0", where_1: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0", gt: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_16: "bf16[8192, 1024][1024, 1]cuda:0", addmm_3: "bf16[8192, 1024][1024, 1]cuda:0", gt_1: "b8[64, 128, 1024][131072, 1024, 1]cuda:0", getitem_3: "f32[64, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[64, 128, 1][128, 1, 1]cuda:0", view_18: "bf16[8192, 1024][1024, 1]cuda:0", view_20: "bf16[8192, 4096][4096, 1]cuda:0", gt_2: "b8[64, 128, 1024][131072, 1024, 1]cuda:0", permute_11: "bf16[1024, 4096][4096, 1]cuda:0", le: "b8[64, 128, 4096][524288, 4096, 1]cuda:0", permute_15: "bf16[4096, 1024][1024, 1]cuda:0", permute_19: "bf16[1024, 1024][1024, 1]cuda:0", permute_24: "bf16[1024, 128, 128][16384, 1, 128]cuda:0", permute_25: "bf16[1024, 64, 128][8192, 1, 64]cuda:0", permute_26: "bf16[1024, 64, 128][8192, 1, 64]cuda:0", permute_27: "bf16[1024, 128, 64][8192, 1, 128]cuda:0", permute_31: "bf16[1024, 1024][1024, 1]cuda:0", permute_35: "bf16[1024, 1024][1024, 1]cuda:0", permute_40: "bf16[1024, 1024][1024, 1]cuda:0", tangents_1: "f32[64, 128, 1024][131072, 1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_40: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:376 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_41: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_12: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_13: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, mul_12);  convert_element_type_40 = mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [8192, 1024]);  mul_13 = None
        mm: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        permute_12: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_1: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
        sum_2: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        convert_element_type_46: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [64, 128, 4096]);  mm = None
        convert_element_type_47: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_48: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        where_2: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.where.self(le, full_default_1, view_24);  le = full_default_1 = view_24 = None
        view_25: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(where_2, [8192, 4096]);  where_2 = None
        mm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
        permute_16: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
        sum_3: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_25, [0], True, dtype = torch.float32);  view_25 = None
        view_26: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        convert_element_type_53: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        view_27: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [64, 128, 1024]);  mm_2 = None
        convert_element_type_54: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        convert_element_type_55: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_56: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_53, torch.float32);  convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_15: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_54, primals_13);  primals_13 = None
        mul_16: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 1024)
        sum_4: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_17: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [64, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:368 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_6: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_17);  view_17 = None
        mul_7: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_3: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_7);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_2: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_8: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_17: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, mul_8);  mul_15 = None
        sum_5: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, sum_5);  sum_5 = None
        sub_4: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_16, sum_4);  mul_16 = sum_4 = None
        sub_5: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_4, mul_18);  sub_4 = mul_18 = None
        div_1: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_19: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_20: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_54, mul_8);  mul_8 = None
        sum_6: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_7: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_54, [0, 1]);  convert_element_type_54 = None
        add_7: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_19);  tangents_1 = mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_57: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:368 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_58: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_21: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_22: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, mul_21);  convert_element_type_57 = mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [8192, 1024]);  mul_22 = None
        mm_4: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
        permute_20: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_5: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
        sum_8: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
        convert_element_type_63: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [64, 128, 1024]);  mm_4 = None
        convert_element_type_64: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_65: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_63, torch.float32);  convert_element_type_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_31: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [64, 128, 16, 64]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_32: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [1024, 128, 64]);  clone_7 = None
        bmm_2: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
        bmm_3: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
        view_33: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 128, 64]);  bmm_2 = None
        view_34: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [64, 16, 128, 128]);  bmm_3 = None
        convert_element_type_70: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_23: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_24: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_34, mul_23);  view_34 = mul_23 = None
        convert_element_type_71: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.float32);  mul_24 = None
        convert_element_type_72: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        mul_25: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_71, convert_element_type_72);  convert_element_type_71 = None
        sum_9: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_25, [-1], True)
        neg: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_72);  convert_element_type_72 = None
        fma: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_9, mul_25);  neg = sum_9 = mul_25 = None
        convert_element_type_73: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_35: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [1024, 128, 128]);  convert_element_type_73 = None
        bmm_4: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_26, view_35);  permute_26 = None
        bmm_5: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_35, permute_27);  view_35 = permute_27 = None
        view_36: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [64, 16, 64, 128]);  bmm_4 = None
        view_37: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 128, 64]);  bmm_5 = None
        mul_26: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_36, 0.3535533905932738);  view_36 = None
        permute_28: "bf16[64, 16, 128, 64][131072, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(mul_26, [0, 1, 3, 2]);  mul_26 = None
        mul_27: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_37, 0.3535533905932738);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_29: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_9: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_38: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [64, 128, 1024]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_30: "bf16[64, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_39: "bf16[64, 128, 1024][131072, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_30, [64, 128, 1024]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [8192, 1024]);  view_38 = None
        mm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
        permute_32: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        sum_10: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
        convert_element_type_82: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [64, 128, 1024]);  mm_6 = None
        convert_element_type_83: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_42, torch.float32);  view_42 = None
        convert_element_type_84: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_85: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        clone_10: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_39, memory_format = torch.contiguous_format);  view_39 = None
        view_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [8192, 1024]);  clone_10 = None
        mm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_43, permute_35);  permute_35 = None
        permute_36: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_43, [1, 0])
        mm_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = None
        sum_11: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_43, [0], True, dtype = torch.float32);  view_43 = None
        view_44: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
        convert_element_type_90: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_44, torch.bfloat16);  view_44 = None
        view_45: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [64, 128, 1024]);  mm_8 = None
        convert_element_type_91: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        add_8: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_83, convert_element_type_91);  convert_element_type_83 = convert_element_type_91 = None
        convert_element_type_92: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_93: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_39: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(mul_27, [0, 2, 1, 3]);  mul_27 = None
        clone_11: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_46: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [64, 128, 1024]);  clone_11 = None
        view_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_46, [8192, 1024]);  view_46 = None
        mm_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
        permute_41: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_12: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_47, [0], True, dtype = torch.float32);  view_47 = None
        view_48: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        convert_element_type_98: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        view_49: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [64, 128, 1024]);  mm_10 = None
        convert_element_type_99: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        add_9: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_8, convert_element_type_99);  add_8 = convert_element_type_99 = None
        convert_element_type_100: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_101: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_98, torch.float32);  convert_element_type_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_29: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, primals_1);  primals_1 = None
        mul_30: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, 1024)
        sum_13: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True)
        sub: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_31: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, mul);  mul_29 = None
        sum_14: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_31, [2], True);  mul_31 = None
        mul_32: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
        sub_7: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_30, sum_13);  mul_30 = sum_13 = None
        sub_8: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_7, mul_32);  sub_7 = mul_32 = None
        div_2: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_33: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_34: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, mul);  mul = None
        sum_15: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_34, [0, 1]);  mul_34 = None
        sum_16: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        add_10: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, mul_33);  add_7 = mul_33 = None
        return (sum_15, sum_16, add_10, convert_element_type_100, convert_element_type_101, convert_element_type_92, convert_element_type_93, convert_element_type_84, convert_element_type_85, None, convert_element_type_64, convert_element_type_65, sum_6, sum_7, convert_element_type_55, convert_element_type_56, convert_element_type_47, convert_element_type_48)
