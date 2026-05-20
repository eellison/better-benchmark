class GraphModule(torch.nn.Module):
    def forward(self, mm_314: "f32[4096, 512]", primals_84: "f32[512]", add_74: "f32[32, 128, 512]", rsqrt_18: "f32[32, 128, 1]", add_257: "f32[32, 128, 512]", gt_37: "b8[32, 128, 512]", primals_83: "f32[512, 384]", mm_424: "f32[4096, 512]", mm_426: "f32[4096, 512]", primals_9: "f32[512]", add_9: "f32[32, 128, 512]", rsqrt_1: "f32[32, 128, 1]", add_337: "f32[32, 128, 512]", gt_3: "b8[32, 128, 512]", primals_8: "f32[512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_314, _shape_param_0);  mm_314 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_84);  reshape_default = primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, add_74)
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_18);  mul_tensor = None
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_257, mul_tensor_2);  add_257 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 1.0);  add_74 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_4);  add_tensor = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_tensor_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_5);  add_tensor_1 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[384, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_2: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_424, _shape_param_3);  mm_424 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_3: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_426, _shape_param_4);  mm_426 = _shape_param_4 = None
        add_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default_2, reshape_default_3);  reshape_default_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_7: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_9);  add_tensor_2 = primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_7, add_9)
        mul_tensor_9: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_7, rsqrt_1);  mul_tensor_7 = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True);  mul_tensor_8 = None
        add_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_337, mul_tensor_9);  add_337 = mul_tensor_9 = None
        pow_tensor_scalar_2: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_scalar_2: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_10: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default_1: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_10, _shape_param_5);  mul_tensor_10 = _shape_param_5 = None
        div_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0);  add_9 = None
        mul_scalar_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_11: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_11);  add_tensor_3 = mul_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_default_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor_12: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_13: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor_12);  add_tensor_4 = mul_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default_4: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_13, _shape_param_6);  mul_tensor_13 = _shape_param_6 = None
        permute_default_2: "f32[384, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_3: "f32[512, 384]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_1, permute_default_1, reshape_default_4, permute_default_3)
