class GraphModule(torch.nn.Module):
    def forward(self, addmm_360: "f32[32768, 512]", add_349: "f32[256, 128, 512]", arg1111_1: "f32[512]", arg1112_1: "f32[512]", arg1113_1: "f32[512, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_360, _shape_param_0);  addmm_360 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default, add_349);  reshape_default = add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, arg1111_1);  add_tensor = arg1111_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor, arg1112_1);  mul_tensor = arg1112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[512, 512]" = torch.ops.aten.permute.default(arg1113_1, [1, 0]);  arg1113_1 = None
        return (reshape_default_1, permute_default)
