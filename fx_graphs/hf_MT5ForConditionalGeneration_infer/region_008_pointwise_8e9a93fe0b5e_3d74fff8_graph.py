class GraphModule(torch.nn.Module):
    def forward(self, mm_133: "f32[4096, 384]", mm_134: "f32[4096, 384]", mul_155: "f32[32, 128, 512]", arg179_1: "f32[384, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_133, _shape_param_0);  mm_133 = _shape_param_0 = None
        reshape_default_1: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # No stacktrace found for following nodes
        permute_default: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        mul_scalar: "f32[32, 6, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
        expand_default: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_2);  mul_scalar = _shape_param_2 = None
        clone_default: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_134, _shape_param_4);  mm_134 = _shape_param_4 = None
        reshape_default_4: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None

        # No stacktrace found for following nodes
        permute_default_1: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        permute_default_2: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[32, 6, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_2, 1.0);  permute_default_2 = None
        expand_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_6: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_155, _shape_param_8);  mul_155 = _shape_param_8 = None
        permute_default_3: "f32[512, 384]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        return (reshape_default_2, reshape_default_5, reshape_default_6, permute_default_3)
