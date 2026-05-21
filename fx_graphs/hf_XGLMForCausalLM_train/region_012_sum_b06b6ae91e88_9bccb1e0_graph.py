class GraphModule(torch.nn.Module):
    def forward(self, view_24: "f32[4096, 1024]", view_27: "f32[4096, 4096]", view_29: "f32[32, 128, 1024]", mul_7: "f32[32, 128, 1024]", view_30: "f32[4096, 1024]", view_46: "f32[4096, 1024]", mm_6: "f32[4096, 1024]", view_49: "f32[4096, 1024]", mm_8: "f32[4096, 1024]", view_52: "f32[4096, 1024]", mm_10: "f32[4096, 1024]", primals_1: "f32[1024]", primals_3: "f32[32, 128, 1024]", getitem_1: "f32[32, 128, 1]", rsqrt: "f32[32, 128, 1]", add_10: "f32[32, 128, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_24, [1, 0])
        sum_dim_int_list: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_24, [0], True);  view_24 = None
        reshape_default: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_27, [1, 0])
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_27, [0], True);  view_27 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_29, mul_7);  mul_7 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_29, [0, 1]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_30, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_30, [0], True);  view_30 = None
        reshape_default_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        permute_default_3: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_46, [1, 0])
        sum_dim_int_list_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_46, [0], True);  view_46 = None
        reshape_default_3: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_3);  sum_dim_int_list_5 = _shape_param_3 = None
        reshape_default_4: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_6, _shape_param_4);  mm_6 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        permute_default_4: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_49, [1, 0])
        sum_dim_int_list_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_49, [0], True);  view_49 = None
        reshape_default_5: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_5);  sum_dim_int_list_6 = _shape_param_5 = None
        reshape_default_6: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_8, _shape_param_6);  mm_8 = _shape_param_6 = None
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default_4, reshape_default_6);  reshape_default_4 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        permute_default_5: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_52, [1, 0])
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        reshape_default_7: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        reshape_default_8: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_10, _shape_param_8);  mm_10 = _shape_param_8 = None
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_8);  add_tensor = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_1);  primals_1 = None
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1024)
        sum_dim_int_list_8: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        sub_tensor: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        sum_dim_int_list_9: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_9);  sum_dim_int_list_9 = None
        sub_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list_8);  mul_tensor_2 = sum_dim_int_list_8 = None
        sub_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_tensor_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_10, mul_tensor_6);  add_10 = mul_tensor_6 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_2, sum_dim_int_list_3, permute_default_2, reshape_default_2, permute_default_3, reshape_default_3, permute_default_4, reshape_default_5, permute_default_5, reshape_default_7, sum_dim_int_list_10, sum_dim_int_list_11, add_tensor_2)
