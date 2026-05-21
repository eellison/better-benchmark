class GraphModule(torch.nn.Module):
    def forward(self, view_433: "f32[8192, 32128]", bmm_80: "f32[64, 1024, 64]", bmm_82: "f32[64, 64, 1024]", bmm_83: "f32[64, 1024, 64]", primals_57: "f32[512, 512]", primals_56: "f32[512, 512]", primals_55: "f32[512, 512]", bmm_104: "f32[64, 1024, 64]", bmm_106: "f32[64, 64, 1024]", bmm_107: "f32[64, 1024, 64]", primals_6: "f32[512, 512]", primals_5: "f32[512, 512]", primals_4: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        permute_default: "f32[32128, 8192]" = torch.ops.aten.permute.default(view_433, [1, 0]);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_80, _shape_param_0);  bmm_80 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_82, _shape_param_1);  bmm_82 = _shape_param_1 = None
        reshape_default_2: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_83, _shape_param_2);  bmm_83 = _shape_param_2 = None
        permute_default_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_3: "f32[512, 512]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_default_4: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_5: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        reshape_default_5: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        clone_default_1: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_6: "f32[512, 512]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_default_7: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_8: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_7: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_default_10: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_9: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_104, _shape_param_9);  bmm_104 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_10: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_106, _shape_param_10);  bmm_106 = _shape_param_10 = None
        reshape_default_11: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_107, _shape_param_11);  bmm_107 = _shape_param_11 = None
        permute_default_11: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_10, [0, 1, 3, 2]);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_12: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_9, [0, 2, 1, 3]);  reshape_default_9 = None
        clone_default_3: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_12, memory_format = torch.contiguous_format);  permute_default_12 = None
        reshape_default_12: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_12);  clone_default_3 = _shape_param_12 = None
        reshape_default_13: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_12, _shape_param_13);  reshape_default_12 = _shape_param_13 = None
        permute_default_13: "f32[512, 512]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_14: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_13, [1, 0]);  permute_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_11, [0, 2, 1, 3]);  permute_default_11 = None
        reshape_default_14: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_15, _shape_param_14);  permute_default_15 = _shape_param_14 = None
        clone_default_4: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_14, memory_format = torch.contiguous_format);  reshape_default_14 = None
        reshape_default_15: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_15);  clone_default_4 = _shape_param_15 = None
        permute_default_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_17: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_16, [1, 0]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_18: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_11, [0, 2, 1, 3]);  reshape_default_11 = None
        clone_default_5: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_18, memory_format = torch.contiguous_format);  permute_default_18 = None
        reshape_default_16: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_16);  clone_default_5 = _shape_param_16 = None
        reshape_default_17: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_17);  reshape_default_16 = _shape_param_17 = None
        permute_default_19: "f32[512, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_20: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_19, [1, 0]);  permute_default_19 = None
        return (permute_default, reshape_default_4, permute_default_4, reshape_default_6, permute_default_7, reshape_default_8, permute_default_10, reshape_default_13, permute_default_14, reshape_default_15, permute_default_17, reshape_default_17, permute_default_20)
