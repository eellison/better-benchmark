class GraphModule(torch.nn.Module):
    def forward(self, bmm_136: "f32[512, 128, 128]", bmm_138: "f32[512, 128, 128]", bmm_139: "f32[512, 128, 128]", primals_22: "f32[2048, 2048]", primals_21: "f32[2048, 2048]", primals_20: "f32[2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        reshape_default: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_136, _shape_param_0);  bmm_136 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_138, _shape_param_1);  bmm_138 = _shape_param_1 = None
        reshape_default_2: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_139, _shape_param_2);  bmm_139 = _shape_param_2 = None
        permute_default: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_default: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_2: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_4: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_3: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_5: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_6: "f32[4096, 2048]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_6);  reshape_default_3 = _shape_param_6 = None
        permute_default_4: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_default_5: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_default_2: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_7: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        permute_default_6: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_default_7: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        reshape_default_8: "f32[4096, 2048]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_8);  reshape_default_5 = _shape_param_8 = None
        permute_default_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_default_9: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_6, permute_default_5, reshape_default_7, permute_default_7, reshape_default_8, permute_default_9)
