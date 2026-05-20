class GraphModule(torch.nn.Module):
    def forward(self, view_607: "f32[4096, 250112]", bmm_108: "f32[192, 128, 64]", bmm_110: "f32[192, 64, 128]", bmm_111: "f32[192, 128, 64]", primals_81: "f32[384, 512]", primals_80: "f32[384, 512]", primals_79: "f32[384, 512]", bmm_140: "f32[192, 128, 64]", bmm_142: "f32[192, 64, 128]", bmm_143: "f32[192, 128, 64]", primals_6: "f32[384, 512]", primals_5: "f32[384, 512]", primals_4: "f32[384, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        permute_default: "f32[250112, 4096]" = torch.ops.aten.permute.default(view_607, [1, 0]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_108, _shape_param_0);  bmm_108 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_110, _shape_param_1);  bmm_110 = _shape_param_1 = None
        reshape_default_2: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_111, _shape_param_2);  bmm_111 = _shape_param_2 = None
        permute_default_1: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_3: "f32[512, 384]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        permute_default_4: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_5: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        reshape_default_5: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        clone_default_1: "f32[32, 128, 384]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_6: "f32[512, 384]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_default_7: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_8: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_7: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_9: "f32[512, 384]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_default_10: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_9: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_9);  bmm_140 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_10: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_10);  bmm_142 = _shape_param_10 = None
        reshape_default_11: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_11);  bmm_143 = _shape_param_11 = None
        permute_default_11: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_10, [0, 1, 3, 2]);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_12: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_9, [0, 2, 1, 3]);  reshape_default_9 = None
        clone_default_3: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_12, memory_format = torch.contiguous_format);  permute_default_12 = None
        reshape_default_12: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_12);  clone_default_3 = _shape_param_12 = None
        reshape_default_13: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_12, _shape_param_13);  reshape_default_12 = _shape_param_13 = None
        permute_default_13: "f32[512, 384]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_14: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_13, [1, 0]);  permute_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_11, [0, 2, 1, 3]);  permute_default_11 = None
        reshape_default_14: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_default_15, _shape_param_14);  permute_default_15 = _shape_param_14 = None
        clone_default_4: "f32[32, 128, 384]" = torch.ops.aten.clone.default(reshape_default_14, memory_format = torch.contiguous_format);  reshape_default_14 = None
        reshape_default_15: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_15);  clone_default_4 = _shape_param_15 = None
        permute_default_16: "f32[512, 384]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_17: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_16, [1, 0]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_18: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_11, [0, 2, 1, 3]);  reshape_default_11 = None
        clone_default_5: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_18, memory_format = torch.contiguous_format);  permute_default_18 = None
        reshape_default_16: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_16);  clone_default_5 = _shape_param_16 = None
        reshape_default_17: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_17);  reshape_default_16 = _shape_param_17 = None
        permute_default_19: "f32[512, 384]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_20: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_19, [1, 0]);  permute_default_19 = None
        return (permute_default, reshape_default_4, permute_default_4, reshape_default_6, permute_default_7, reshape_default_8, permute_default_10, reshape_default_13, permute_default_14, reshape_default_15, permute_default_17, reshape_default_17, permute_default_20)
