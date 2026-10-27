class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024][1]cuda:0", primals_3: "f32[64, 128, 1024][131072, 1024, 1]cuda:0", primals_14: "f32[1024][1]cuda:0", primals_25: "f32[1024][1]cuda:0", getitem_1: "f32[64, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[64, 128, 1][128, 1, 1]cuda:0", view: "bf16[8192, 1024][1024, 1]cuda:0", permute_1: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0", permute_4: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0", permute_5: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0", where: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0", getitem_2: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0", getitem_3: "f32[64, 16, 128][2048, 128, 1]cuda:0", getitem_4: "i64[][]cuda:0", getitem_5: "i64[][]cuda:0", addmm_3: "bf16[8192, 1024][1024, 1]cuda:0", gt: "b8[64, 128, 1024][131072, 1024, 1]cuda:0", getitem_7: "f32[64, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[64, 128, 1][128, 1, 1]cuda:0", view_12: "bf16[8192, 1024][1024, 1]cuda:0", view_15: "bf16[8192, 1024][1024, 1]cuda:0", where_2: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0", gt_1: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_28: "bf16[8192, 1024][1024, 1]cuda:0", gt_2: "b8[64, 128, 1024][131072, 1024, 1]cuda:0", mul_12: "f32[64, 128, 1024][131072, 1024, 1]cuda:0", view_30: "bf16[8192, 1024][1024, 1]cuda:0", view_32: "bf16[8192, 4096][4096, 1]cuda:0", gt_3: "b8[64, 128, 1024][131072, 1024, 1]cuda:0", permute_19: "bf16[1024, 4096][4096, 1]cuda:0", le: "b8[64, 128, 4096][524288, 4096, 1]cuda:0", permute_23: "bf16[4096, 1024][1024, 1]cuda:0", div_1: "f32[64, 128, 1][128, 1, 1]cuda:0", permute_27: "bf16[1024, 1024][1024, 1]cuda:0", permute_32: "bf16[1024, 128, 128][16384, 1, 128]cuda:0", permute_33: "bf16[1024, 64, 128][8192, 1, 64]cuda:0", permute_34: "bf16[1024, 64, 128][8192, 1, 64]cuda:0", permute_35: "bf16[1024, 128, 64][8192, 1, 128]cuda:0", permute_39: "bf16[1024, 1024][1024, 1]cuda:0", permute_43: "bf16[1024, 1024][1024, 1]cuda:0", permute_48: "bf16[1024, 1024][1024, 1]cuda:0", permute_52: "bf16[1024, 1024][1024, 1]cuda:0", permute_59: "bf16[1024, 1024][1024, 1]cuda:0", permute_63: "bf16[1024, 1024][1024, 1]cuda:0", permute_68: "bf16[1024, 1024][1024, 1]cuda:0", tangents_1: "f32[64, 128, 1024][131072, 1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_63: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:474 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_64: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_16: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_17: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_63, mul_16);  convert_element_type_63 = mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_34: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_17, [8192, 1024]);  mul_17 = None
        mm: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_34, permute_19);  permute_19 = None
        permute_20: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_1: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_20, view_32);  permute_20 = view_32 = None
        sum_2: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_34, [0], True, dtype = torch.float32);  view_34 = None
        view_35: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        convert_element_type_69: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.bfloat16);  view_35 = None
        view_36: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [64, 128, 4096]);  mm = None
        convert_element_type_70: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_71: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_69, torch.float32);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        where_3: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.where.self(le, full_default_1, view_36);  le = full_default_1 = view_36 = None
        view_37: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(where_3, [8192, 4096]);  where_3 = None
        mm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_37, permute_23);  permute_23 = None
        permute_24: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_24, view_30);  permute_24 = view_30 = None
        sum_3: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        view_38: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        convert_element_type_76: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        view_39: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [64, 128, 1024]);  mm_2 = None
        convert_element_type_77: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        convert_element_type_78: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_79: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_76, torch.float32);  convert_element_type_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_19: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, primals_25);  primals_25 = None
        mul_20: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, 1024)
        sum_4: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_19, [2], True)
        mul_21: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, mul_12);  mul_19 = None
        sum_5: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True);  mul_21 = None
        mul_22: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, sum_5);  sum_5 = None
        sub_5: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_20, sum_4);  mul_20 = sum_4 = None
        sub_6: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_5, mul_22);  sub_5 = mul_22 = None
        mul_23: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_6);  div_1 = sub_6 = None
        mul_24: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, mul_12);  mul_12 = None
        sum_6: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_24, [0, 1]);  mul_24 = None
        sum_7: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_77, [0, 1]);  convert_element_type_77 = None
        add_10: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_23);  tangents_1 = mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_80: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:465 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_81: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_25: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
        mul_26: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, mul_25);  convert_element_type_80 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_26, [8192, 1024]);  mul_26 = None
        mm_4: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_27);  permute_27 = None
        permute_28: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_5: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_28, view_28);  permute_28 = view_28 = None
        sum_8: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
        convert_element_type_86: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [64, 128, 1024]);  mm_4 = None
        convert_element_type_87: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_88: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_86, torch.float32);  convert_element_type_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_43: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [64, 128, 16, 64]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_44: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [1024, 128, 64]);  clone_7 = None
        bmm_2: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_32, view_44);  permute_32 = None
        bmm_3: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_44, permute_33);  view_44 = permute_33 = None
        view_45: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 128, 64]);  bmm_2 = None
        view_46: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [64, 16, 128, 128]);  bmm_3 = None
        convert_element_type_93: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_27: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.1111111111111112);  convert_element_type_93 = None
        mul_28: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_46, mul_27);  view_46 = mul_27 = None
        convert_element_type_94: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.float32);  mul_28 = None
        convert_element_type_95: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        mul_29: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, convert_element_type_95);  convert_element_type_94 = None
        sum_9: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_29, [-1], True)
        neg: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_95);  convert_element_type_95 = None
        fma: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_9, mul_29);  neg = sum_9 = mul_29 = None
        convert_element_type_96: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_47: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_96, [1024, 128, 128]);  convert_element_type_96 = None
        bmm_4: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_34, view_47);  permute_34 = None
        bmm_5: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_47, permute_35);  view_47 = permute_35 = None
        view_48: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [64, 16, 64, 128]);  bmm_4 = None
        view_49: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 128, 64]);  bmm_5 = None
        mul_30: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_48, 0.3535533905932738);  view_48 = None
        permute_36: "bf16[64, 16, 128, 64][131072, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(mul_30, [0, 1, 3, 2]);  mul_30 = None
        mul_31: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_49, 0.3535533905932738);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_37: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_9: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_50: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [64, 128, 1024]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_38: "bf16[64, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_36, [0, 2, 1, 3]);  permute_36 = None
        view_51: "bf16[64, 128, 1024][131072, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_38, [64, 128, 1024]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_52: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [8192, 1024]);  view_50 = None
        mm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_52, permute_39);  permute_39 = None
        permute_40: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_40, view_15);  permute_40 = None
        sum_10: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_52, [0], True, dtype = torch.float32);  view_52 = None
        view_53: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
        convert_element_type_105: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.bfloat16);  view_53 = None
        view_54: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [64, 128, 1024]);  mm_6 = None
        convert_element_type_106: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_54, torch.float32);  view_54 = None
        convert_element_type_107: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_108: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_105, torch.float32);  convert_element_type_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        clone_10: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_51, memory_format = torch.contiguous_format);  view_51 = None
        view_55: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [8192, 1024]);  clone_10 = None
        mm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_55, permute_43);  permute_43 = None
        permute_44: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_44, view_15);  permute_44 = view_15 = None
        sum_11: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_55, [0], True, dtype = torch.float32);  view_55 = None
        view_56: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
        convert_element_type_113: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.bfloat16);  view_56 = None
        view_57: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [64, 128, 1024]);  mm_8 = None
        convert_element_type_114: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        add_11: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_106, convert_element_type_114);  convert_element_type_106 = convert_element_type_114 = None
        convert_element_type_115: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_116: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_113, torch.float32);  convert_element_type_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_47: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(mul_31, [0, 2, 1, 3]);  mul_31 = None
        clone_11: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_58: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [64, 128, 1024]);  clone_11 = None
        view_59: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_58, [8192, 1024]);  view_58 = None
        mm_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_59, permute_48);  permute_48 = None
        permute_49: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_59, [1, 0])
        mm_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_49, view_12);  permute_49 = view_12 = None
        sum_12: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_59, [0], True, dtype = torch.float32);  view_59 = None
        view_60: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        convert_element_type_121: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_60, torch.bfloat16);  view_60 = None
        view_61: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [64, 128, 1024]);  mm_10 = None
        convert_element_type_122: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        convert_element_type_123: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_124: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        mul_33: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, primals_14);  primals_14 = None
        mul_34: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, 1024)
        sum_13: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [64, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:450 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_1: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_7);  add_2 = getitem_7 = None
        mul_4: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_35: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, mul_4);  mul_33 = None
        sum_14: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, sum_14);  sum_14 = None
        sub_8: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_34, sum_13);  mul_34 = sum_13 = None
        sub_9: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        div_2: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_37: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_9);  div_2 = sub_9 = None
        mul_38: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, mul_4);  mul_4 = None
        sum_15: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_16: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_122, [0, 1]);  convert_element_type_122 = None
        add_12: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, mul_37);  add_10 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_125: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:450 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_126: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_39: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, 1.1111111111111112);  convert_element_type_126 = None
        mul_40: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, mul_39);  convert_element_type_125 = mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [8192, 1024]);  mul_40 = None
        mm_12: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_62, permute_52);  permute_52 = None
        permute_53: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_62, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [64, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        mm_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_53, view_10);  permute_53 = view_10 = None
        sum_17: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_62, [0], True, dtype = torch.float32);  view_62 = None
        view_63: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_17, [1024]);  sum_17 = None
        convert_element_type_131: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        view_64: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [64, 128, 1024]);  mm_12 = None
        convert_element_type_132: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_133: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_131, torch.float32);  convert_element_type_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_65: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [64, 128, 16, 64]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "bf16[64, 16, 128, 128][16384, 0, 128, 1]cuda:0" = torch.ops.aten.expand.default(where, [64, 16, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_56, permute_1, permute_4, permute_5, expand, getitem_2, getitem_3, getitem_4, getitem_5, 0.1, [True, True, True, False], scale = 0.125);  permute_56 = permute_1 = permute_4 = permute_5 = expand = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_10: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_11: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_12: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_57: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        view_66: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [64, 128, 1024]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_58: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
        view_67: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_58, [64, 128, 1024]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [8192, 1024]);  view_66 = None
        mm_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_68, permute_59);  permute_59 = None
        permute_60: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_68, [1, 0])
        mm_15: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_60, view);  permute_60 = None
        sum_18: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_68, [0], True, dtype = torch.float32);  view_68 = None
        view_69: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [1024]);  sum_18 = None
        convert_element_type_138: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.bfloat16);  view_69 = None
        view_70: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [64, 128, 1024]);  mm_14 = None
        convert_element_type_139: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        convert_element_type_140: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_141: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_138, torch.float32);  convert_element_type_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_71: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [8192, 1024]);  view_67 = None
        mm_16: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_71, permute_63);  permute_63 = None
        permute_64: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_71, [1, 0])
        mm_17: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_64, view);  permute_64 = None
        sum_19: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_71, [0], True, dtype = torch.float32);  view_71 = None
        view_72: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [1024]);  sum_19 = None
        convert_element_type_146: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_72, torch.bfloat16);  view_72 = None
        view_73: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [64, 128, 1024]);  mm_16 = None
        convert_element_type_147: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        add_13: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_139, convert_element_type_147);  convert_element_type_139 = convert_element_type_147 = None
        convert_element_type_148: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_149: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_146, torch.float32);  convert_element_type_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_74: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_67, [64, 128, 1024]);  permute_67 = None
        view_75: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [8192, 1024]);  view_74 = None
        mm_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_75, permute_68);  permute_68 = None
        permute_69: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_19: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_69, view);  permute_69 = view = None
        sum_20: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_75, [0], True, dtype = torch.float32);  view_75 = None
        view_76: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_20, [1024]);  sum_20 = None
        convert_element_type_154: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_76, torch.bfloat16);  view_76 = None
        view_77: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [64, 128, 1024]);  mm_18 = None
        convert_element_type_155: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        add_14: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, convert_element_type_155);  add_13 = convert_element_type_155 = None
        convert_element_type_156: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_157: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_154, torch.float32);  convert_element_type_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_42: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, primals_1);  primals_1 = None
        mul_43: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, 1024)
        sum_21: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_42, [2], True)
        sub: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_44: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, mul);  mul_42 = None
        sum_22: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_44, [2], True);  mul_44 = None
        mul_45: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_22);  sum_22 = None
        sub_11: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_43, sum_21);  mul_43 = sum_21 = None
        sub_12: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_11, mul_45);  sub_11 = mul_45 = None
        div_3: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_46: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_12);  div_3 = sub_12 = None
        mul_47: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, mul);  mul = None
        sum_23: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_47, [0, 1]);  mul_47 = None
        sum_24: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        add_15: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_12, mul_46);  add_12 = mul_46 = None
        return (sum_23, sum_24, add_15, convert_element_type_156, convert_element_type_157, convert_element_type_148, convert_element_type_149, convert_element_type_140, convert_element_type_141, None, convert_element_type_132, convert_element_type_133, add_11, sum_15, sum_16, convert_element_type_123, convert_element_type_124, convert_element_type_115, convert_element_type_116, convert_element_type_107, convert_element_type_108, None, convert_element_type_87, convert_element_type_88, sum_6, sum_7, convert_element_type_78, convert_element_type_79, convert_element_type_70, convert_element_type_71)
