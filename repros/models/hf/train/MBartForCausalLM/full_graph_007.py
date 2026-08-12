class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024][1]cuda:0", primals_3: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0", primals_13: "f32[1024][1]cuda:0", getitem_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0", rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view: "bf16[8192, 1024][1024, 1]cuda:0", permute_1: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0", permute_4: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0", permute_5: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0", where: "bf16[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0", getitem_2: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0", getitem_3: "f32[8, 16, 1024, 1][16384, 1024, 1, 1]cuda:0", getitem_8: "i64[][]cuda:0", getitem_9: "i64[][]cuda:0", addmm_3: "bf16[8192, 1024][1024, 1]cuda:0", gt: "b8[8, 1024, 1024][1048576, 1024, 1]cuda:0", getitem_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0", rsqrt_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_12: "bf16[8192, 1024][1024, 1]cuda:0", addmm_4: "bf16[8192, 4096][4096, 1]cuda:0", view_14: "bf16[8192, 4096][4096, 1]cuda:0", gt_1: "b8[8, 1024, 1024][1048576, 1024, 1]cuda:0", permute_10: "bf16[1024, 4096][4096, 1]cuda:0", permute_14: "bf16[4096, 1024][1024, 1]cuda:0", permute_18: "bf16[1024, 1024][1024, 1]cuda:0", permute_25: "bf16[1024, 1024][1024, 1]cuda:0", permute_29: "bf16[1024, 1024][1024, 1]cuda:0", permute_34: "bf16[1024, 1024][1024, 1]cuda:0", tangents_1: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_36: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:416 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_37: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_11: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_12: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, mul_11);  convert_element_type_36 = mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_16: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [8192, 1024]);  mul_12 = None
        mm: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_16, permute_10);  permute_10 = None
        permute_11: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_1: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_11, view_14);  permute_11 = view_14 = None
        sum_1: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_16, [0], True, dtype = torch.float32);  view_16 = None
        view_17: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1024]);  sum_1 = None
        convert_element_type_42: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_17, torch.bfloat16);  view_17 = None
        view_18: "bf16[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [8, 1024, 4096]);  mm = None
        convert_element_type_43: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_44: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_42, torch.float32);  convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_45: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_13: "bf16[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_29: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_7: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, 0.7071067811865476)
        erf: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_5: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
        mul_15: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, convert_element_type_29)
        mul_16: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, -0.5);  mul_15 = None
        exp: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_16);  mul_16 = None
        mul_17: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_18: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, mul_17);  convert_element_type_29 = mul_17 = None
        add_8: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, mul_18);  mul_14 = mul_18 = None
        mul_19: "f32[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, add_8);  convert_element_type_45 = add_8 = None
        convert_element_type_47: "bf16[8, 1024, 4096][4194304, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_47, [8192, 4096]);  convert_element_type_47 = None
        mm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_19, permute_14);  permute_14 = None
        permute_15: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_15, view_12);  permute_15 = view_12 = None
        sum_2: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_19, [0], True, dtype = torch.float32);  view_19 = None
        view_20: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [4096]);  sum_2 = None
        convert_element_type_52: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_20, torch.bfloat16);  view_20 = None
        view_21: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 1024]);  mm_2 = None
        convert_element_type_53: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        convert_element_type_54: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_55: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32);  convert_element_type_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_21: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, primals_13);  primals_13 = None
        mul_22: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, 1024)
        sum_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:392 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_1: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_12);  add_2 = getitem_12 = None
        mul_4: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_23: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, mul_4);  mul_21 = None
        sum_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True);  mul_23 = None
        mul_24: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, sum_4);  sum_4 = None
        sub_3: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_22, sum_3);  mul_22 = sum_3 = None
        sub_4: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_3, mul_24);  sub_3 = mul_24 = None
        div: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_25: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_26: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_4);  mul_4 = None
        sum_5: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_26, [0, 1]);  mul_26 = None
        sum_6: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_53, [0, 1]);  convert_element_type_53 = None
        add_9: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_25);  tangents_1 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_56: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:392 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_57: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_27: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_28: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, mul_27);  convert_element_type_56 = mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_22: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [8192, 1024]);  mul_28 = None
        mm_4: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_18);  permute_18 = None
        permute_19: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_22, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[8, 1024, 16, 64][1048576, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [8, 1024, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        mm_5: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_7: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [1024]);  sum_7 = None
        convert_element_type_62: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [8, 1024, 1024]);  mm_4 = None
        convert_element_type_63: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_64: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_62, torch.float32);  convert_element_type_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "bf16[8, 1024, 16, 64][1048576, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [8, 1024, 16, 64]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_22: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_22, permute_1, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, where, None, None, 1024, 1024, 0.0, False, scale = 0.125);  permute_22 = permute_1 = permute_4 = permute_5 = getitem_2 = getitem_3 = getitem_8 = getitem_9 = where = None
        getitem_13: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_14: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_15: "bf16[8, 16, 1024, 64][1048576, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_23: "bf16[8, 1024, 16, 64][1048576, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        view_26: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_23, [8, 1024, 1024]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_24: "bf16[8, 1024, 16, 64][1048576, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3]);  getitem_14 = None
        view_27: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_24, [8, 1024, 1024]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_28: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [8192, 1024]);  view_26 = None
        mm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_25);  permute_25 = None
        permute_26: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_26, view);  permute_26 = None
        sum_8: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
        convert_element_type_69: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [8, 1024, 1024]);  mm_6 = None
        convert_element_type_70: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        convert_element_type_71: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_72: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_69, torch.float32);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_31: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [8192, 1024]);  view_27 = None
        mm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_31, permute_29);  permute_29 = None
        permute_30: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_30, view);  permute_30 = None
        sum_9: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_31, [0], True, dtype = torch.float32);  view_31 = None
        view_32: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_9, [1024]);  sum_9 = None
        convert_element_type_77: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_32, torch.bfloat16);  view_32 = None
        view_33: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [8, 1024, 1024]);  mm_8 = None
        convert_element_type_78: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        add_10: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_70, convert_element_type_78);  convert_element_type_70 = convert_element_type_78 = None
        convert_element_type_79: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_80: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_77, torch.float32);  convert_element_type_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_33: "bf16[8, 1024, 16, 64][1048576, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
        view_34: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [8, 1024, 1024]);  permute_33 = None
        view_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [8192, 1024]);  view_34 = None
        mm_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_35, permute_34);  permute_34 = None
        permute_35: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_35, view);  permute_35 = view = None
        sum_10: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_35, [0], True, dtype = torch.float32);  view_35 = None
        view_36: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
        convert_element_type_85: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.bfloat16);  view_36 = None
        view_37: "bf16[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [8, 1024, 1024]);  mm_10 = None
        convert_element_type_86: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        add_11: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, convert_element_type_86);  add_10 = convert_element_type_86 = None
        convert_element_type_87: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_88: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_85, torch.float32);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_30: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
        mul_31: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 1024)
        sum_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True)
        sub: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_32: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, mul);  mul_30 = None
        sum_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_32, [2], True);  mul_32 = None
        mul_33: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_12);  sum_12 = None
        sub_6: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_31, sum_11);  mul_31 = sum_11 = None
        sub_7: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_6, mul_33);  sub_6 = mul_33 = None
        div_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_34: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_35: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, mul);  mul = None
        sum_13: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_35, [0, 1]);  mul_35 = None
        sum_14: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        add_12: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, mul_34);  add_9 = mul_34 = None
        return (sum_13, sum_14, add_12, convert_element_type_87, convert_element_type_88, convert_element_type_79, convert_element_type_80, convert_element_type_71, convert_element_type_72, None, convert_element_type_63, convert_element_type_64, sum_5, sum_6, convert_element_type_54, convert_element_type_55, convert_element_type_43, convert_element_type_44)
