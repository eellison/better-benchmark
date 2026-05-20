class GraphModule(torch.nn.Module):
    def forward(self, primals_1068: "f32[128, 512]", bmm_47: "f32[1024, 128, 32]", primals_1082: "f32[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_default: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1068, [1, 0]);  primals_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_47, _shape_param_0);  bmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_1: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_2: "f32[32768, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_2: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1082, [1, 0]);  primals_1082 = None
        return (permute_default, reshape_default_2, permute_default_2)
