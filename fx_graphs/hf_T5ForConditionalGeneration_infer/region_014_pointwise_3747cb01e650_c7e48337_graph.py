class GraphModule(torch.nn.Module):
    def forward(self, mm_36: "f32[8192, 512]", mm_37: "f32[8192, 512]", mul_29: "f32[8, 1024, 512]", arg56_1: "f32[512, 512]", bmm_default_13: "f32[64, 1024, 64]", arg47_1: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_36, _shape_param_0);  mm_36 = _shape_param_0 = None
        reshape_default_1: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # No stacktrace found for following nodes
        permute_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        mul_scalar: "f32[8, 8, 1024, 64]" = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
        expand_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_2);  mul_scalar = _shape_param_2 = None
        clone_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_37, _shape_param_4);  mm_37 = _shape_param_4 = None
        reshape_default_4: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None

        # No stacktrace found for following nodes
        permute_default_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        permute_default_2: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.mul.Scalar(permute_default_2, 1.0);  permute_default_2 = None
        expand_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_6: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_29, _shape_param_8);  mul_29 = _shape_param_8 = None
        permute_default_3: "f32[512, 512]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_7: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_default_13, _shape_param_9);  bmm_default_13 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 2, 1, 3]);  reshape_default_7 = None
        clone_default_2: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_8: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_10);  clone_default_2 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_9: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_11);  reshape_default_8 = _shape_param_11 = None
        permute_default_5: "f32[512, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        return (reshape_default_2, reshape_default_5, reshape_default_6, permute_default_3, reshape_default_9, permute_default_5)
