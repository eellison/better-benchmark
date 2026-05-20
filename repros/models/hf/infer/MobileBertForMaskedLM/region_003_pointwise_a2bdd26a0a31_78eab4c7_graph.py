class GraphModule(torch.nn.Module):
    def forward(self, addmm_359: "f32[32768, 128]", add_360: "f32[256, 128, 128]", arg1107_1: "f32[128]", arg1108_1: "f32[128]", arg1109_1: "f32[512, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_359, _shape_param_0);  addmm_359 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, add_360);  reshape_default = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, arg1107_1);  add_tensor = arg1107_1 = None
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, arg1108_1);  mul_tensor = arg1108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[128, 512]" = torch.ops.aten.permute.default(arg1109_1, [1, 0]);  arg1109_1 = None
        return (reshape_default_1, permute_default)
