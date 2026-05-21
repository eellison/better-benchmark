class GraphModule(torch.nn.Module):
    def forward(self, bmm_107: "f32[192, 128, 64]", primals_85: "f32[384, 512]", mm_422: "f32[4096, 1024]", gt_4: "b8[32, 128, 1024]", mm_4: "f32[4096, 1024]", mm_5: "f32[4096, 1024]", primals_11: "f32[1024, 512]", primals_10: "f32[1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_107, _shape_param_0);  bmm_107 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_default_2: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_3: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_422, _shape_param_3);  mm_422 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_3, mul_tensor);  reshape_default_3 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_4: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, _shape_param_4);  mm_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_4, 0.5)
        pow_tensor_scalar: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_4, 3.0)
        mul_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default_4, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_5: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, _shape_param_5);  mm_5 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, reshape_default_5);  mul_tensor_1 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_6: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_6);  mul_tensor_6 = _shape_param_6 = None
        permute_default_3: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_4: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_1);  mul_tensor_7 = add_tensor_1 = None
        mul_tensor_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_11: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor);  mul_tensor_8 = sub_tensor = None
        mul_tensor_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.7978845608028654);  mul_tensor_11 = None
        mul_tensor_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.044715)
        pow_tensor_scalar_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_4, 2.0);  reshape_default_4 = None
        mul_scalar: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_scalar);  mul_tensor_13 = mul_scalar = None
        add_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_14);  mul_tensor_12 = mul_tensor_14 = None
        mul_tensor_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.5);  mul_tensor_9 = None
        add_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_15);  add_tensor_2 = mul_tensor_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_7: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_7);  add_tensor_3 = _shape_param_7 = None
        permute_default_5: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_6: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None
        return (reshape_default_2, permute_default_2, reshape_default_6, permute_default_4, reshape_default_7, permute_default_6)
