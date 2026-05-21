class GraphModule(torch.nn.Module):
    def forward(self, view_34: "f32[2048, 2560]", view_37: "f32[2048, 10240]", view_39: "f32[16, 128, 2560]", mul_10: "f32[16, 128, 2560]", view_40: "f32[2048, 2560]", view_52: "f32[2048, 2560]", mm_6: "f32[2048, 2560]", view_55: "f32[2048, 2560]", mm_8: "f32[2048, 2560]", view_59: "f32[2048, 2560]", view_61: "f32[16, 128, 2560]", mul_4: "f32[16, 128, 2560]", view_62: "f32[2048, 2560]", getitem_2: "f32[16, 32, 128, 80]", view_68: "f32[2048, 2560]", mm_14: "f32[2048, 2560]", view_71: "f32[2048, 2560]", mm_16: "f32[2048, 2560]", view_75: "f32[2048, 2560]", mm_18: "f32[2048, 2560]", primals_1: "f32[2560]", primals_3: "f32[16, 128, 2560]", getitem_1: "f32[16, 128, 1]", rsqrt: "f32[16, 128, 1]", add_15: "f32[16, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_34, [1, 0])
        sum_dim_int_list: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        reshape_default: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_default_1: "f32[10240, 2048]" = torch.ops.aten.permute.default(view_37, [1, 0])
        sum_dim_int_list_1: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        reshape_default_1: "f32[10240]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_39, mul_10);  mul_10 = None
        sum_dim_int_list_2: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_3: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_39, [0, 1]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_40, [1, 0])
        sum_dim_int_list_4: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        reshape_default_2: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_default_3: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_52, [1, 0])
        sum_dim_int_list_5: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        reshape_default_3: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_3);  sum_dim_int_list_5 = _shape_param_3 = None
        reshape_default_4: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_6, _shape_param_4);  mm_6 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_default_4: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_55, [1, 0])
        sum_dim_int_list_6: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        reshape_default_5: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_5);  sum_dim_int_list_6 = _shape_param_5 = None
        reshape_default_6: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_8, _shape_param_6);  mm_8 = _shape_param_6 = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(reshape_default_4, reshape_default_6);  reshape_default_4 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_59, [1, 0])
        sum_dim_int_list_7: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_59, [0], True);  view_59 = None
        reshape_default_7: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_61, mul_4);  mul_4 = None
        sum_dim_int_list_8: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_9: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_61, [0, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_6: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_62, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_7: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_8: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_8);  permute_default_7 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_9: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        sum_dim_int_list_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True);  view_62 = None
        reshape_default_10: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_default_8: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_68, [1, 0])
        sum_dim_int_list_11: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True);  view_68 = None
        reshape_default_11: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_11);  sum_dim_int_list_11 = _shape_param_11 = None
        reshape_default_12: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_14, _shape_param_12);  mm_14 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_default_9: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_71, [1, 0])
        sum_dim_int_list_12: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True);  view_71 = None
        reshape_default_13: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_13);  sum_dim_int_list_12 = _shape_param_13 = None
        reshape_default_14: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_16, _shape_param_14);  mm_16 = _shape_param_14 = None
        add_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(reshape_default_12, reshape_default_14);  reshape_default_12 = reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_10: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_75, [1, 0])
        sum_dim_int_list_13: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        reshape_default_15: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_15);  sum_dim_int_list_13 = _shape_param_15 = None
        reshape_default_16: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_18, _shape_param_16);  mm_18 = _shape_param_16 = None
        add_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_16);  add_tensor_1 = reshape_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_1);  primals_1 = None
        mul_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 2560)
        sum_dim_int_list_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor_4: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = None
        sum_dim_int_list_15: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_15);  sum_dim_int_list_15 = None
        sub_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_14);  mul_tensor_3 = sum_dim_int_list_14 = None
        sub_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_tensor_7: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_16: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_17: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1]);  add_tensor_2 = None
        add_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_15, mul_tensor_7);  add_15 = mul_tensor_7 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_2, sum_dim_int_list_3, permute_default_2, reshape_default_2, permute_default_3, reshape_default_3, permute_default_4, reshape_default_5, add_tensor, permute_default_5, reshape_default_7, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_6, reshape_default_9, reshape_default_10, permute_default_8, reshape_default_11, permute_default_9, reshape_default_13, permute_default_10, reshape_default_15, sum_dim_int_list_16, sum_dim_int_list_17, add_tensor_3)
