class GraphModule(torch.nn.Module):
    def forward(self, view_139: "f32[32768, 30522]", bmm_32: "f32[3072, 128, 64]", bmm_34: "f32[3072, 64, 128]", bmm_35: "f32[3072, 128, 64]", primals_11: "f32[768, 768]", primals_9: "f32[768, 768]", primals_7: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        permute_default: "f32[30522, 32768]" = torch.ops.aten.permute.default(view_139, [1, 0]);  view_139 = None
        constant_pad_nd_default: "f32[30524, 32768]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 2]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_32, _shape_param_0);  bmm_32 = _shape_param_0 = None
        reshape_default_1: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_34, _shape_param_1);  bmm_34 = _shape_param_1 = None
        reshape_default_2: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, _shape_param_2);  bmm_35 = _shape_param_2 = None
        mul_scalar: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.3535533905932738);  reshape_default_1 = None
        permute_default_1: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(reshape_default_2, 0.3535533905932738);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_2: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[32768, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_4: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_5: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        reshape_default_5: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        clone_default_1: "f32[256, 128, 768]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_7: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_8: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_2: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_7: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[32768, 768]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_10: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None
        return (constant_pad_nd_default, reshape_default_4, permute_default_4, reshape_default_6, permute_default_7, reshape_default_8, permute_default_10)
