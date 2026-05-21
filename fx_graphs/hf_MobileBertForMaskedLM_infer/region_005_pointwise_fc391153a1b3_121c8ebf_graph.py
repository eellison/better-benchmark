class GraphModule(torch.nn.Module):
    def forward(self, addmm_351: "f32[32768, 128]", addmm_346: "f32[32768, 128]", arg1069_1: "f32[128]", arg1070_1: "f32[128]", arg1083_1: "f32[128]", arg1084_1: "f32[128]", arg1085_1: "f32[512, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_351, _shape_param_0);  addmm_351 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_346, _shape_param_1);  addmm_346 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, arg1069_1);  reshape_default_1 = arg1069_1 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, arg1070_1);  mul_tensor = arg1070_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg1083_1);  add_tensor_1 = arg1083_1 = None
        add_tensor_2: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1084_1);  mul_tensor_1 = arg1084_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default: "f32[128, 512]" = torch.ops.aten.permute.default(arg1085_1, [1, 0]);  arg1085_1 = None
        return (reshape_default_2, permute_default)
