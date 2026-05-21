class GraphModule(torch.nn.Module):
    def forward(self, view_16: "f32[8192, 1024]", view_19: "f32[8192, 4096]", view_21: "f32[8, 1024, 1024]", mul_4: "f32[8, 1024, 1024]", view_22: "f32[8192, 1024]", getitem_2: "f32[8, 16, 1024, 64]", view_28: "f32[8192, 1024]", mm_6: "f32[8192, 1024]", view_31: "f32[8192, 1024]", mm_8: "f32[8192, 1024]", view_35: "f32[8192, 1024]", mm_10: "f32[8192, 1024]", primals_1: "f32[1024]", primals_3: "f32[8, 1024, 1024]", getitem_1: "f32[8, 1024, 1]", rsqrt: "f32[8, 1024, 1]", add_9: "f32[8, 1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_16, [1, 0])
        sum_dim_int_list: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        reshape_default: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_default_1: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_19, [1, 0])
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_4);  mul_4 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_21, [0, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_22, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_3: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_2);  permute_default_3 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        reshape_default_4: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        permute_default_4: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_28, [1, 0])
        sum_dim_int_list_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        reshape_default_5: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None
        reshape_default_6: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_6, _shape_param_6);  mm_6 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        permute_default_5: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_31, [1, 0])
        sum_dim_int_list_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_31, [0], True);  view_31 = None
        reshape_default_7: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_7);  sum_dim_int_list_6 = _shape_param_7 = None
        reshape_default_8: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_8, _shape_param_8);  mm_8 = _shape_param_8 = None
        add_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_6, reshape_default_8);  reshape_default_6 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_35, [1, 0])
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True);  view_35 = None
        reshape_default_9: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_9);  sum_dim_int_list_7 = _shape_param_9 = None
        reshape_default_10: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_10, _shape_param_10);  mm_10 = _shape_param_10 = None
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_10);  add_tensor = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_1);  primals_1 = None
        mul_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1024)
        sum_dim_int_list_8: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_4: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        sum_dim_int_list_9: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_9);  sum_dim_int_list_9 = None
        sub_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list_8);  mul_tensor_2 = sum_dim_int_list_8 = None
        sub_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_tensor_6: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_7: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_9, mul_tensor_6);  add_9 = mul_tensor_6 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_2, sum_dim_int_list_3, permute_default_2, reshape_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, reshape_default_7, permute_default_6, reshape_default_9, sum_dim_int_list_10, sum_dim_int_list_11, add_tensor_2)
