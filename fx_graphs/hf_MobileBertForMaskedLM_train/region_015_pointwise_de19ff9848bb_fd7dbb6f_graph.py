class GraphModule(torch.nn.Module):
    def forward(self, mm_715: "f32[32768, 512]", mul_784: "f32[256, 128, 512]", mm_721: "f32[32768, 512]", mm_723: "f32[32768, 512]", primals_8: "f32[512]", primals_4: "f32[512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_715, _shape_param_0);  mm_715 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_784, reshape_default);  mul_784 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_721, _shape_param_1);  mm_721 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None
        reshape_default_2: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_723, _shape_param_2);  mm_723 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_8);  add_tensor_2 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default_3: "f32[32768, 512]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        permute_default: "f32[384, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)
