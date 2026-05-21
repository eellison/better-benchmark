class GraphModule(torch.nn.Module):
    def forward(self, getitem_128: "f32[32, 6, 512, 64]", getitem_129: "f32[32, 6, 512, 64]", getitem_130: "f32[32, 6, 512, 64]", div_33: "f32[98304, 9, 1]", view_462: "f32[98304, 64, 1]", unsqueeze_8: "i64[9, 512, 1, 1]", add_7: "i64[1, 1]", primals_273: "f32[384, 768]", convolution_23: "f32[32, 384, 512]", primals_268: "f32[384, 1]", view_476: "f32[32, 512, 384]", primals_269: "f32[384, 768]", primals_264: "f32[384, 768]", primals_262: "f32[384, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3]);  getitem_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_2: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default: "f32[98304, 9, 1]" = torch.ops.aten.expand.default(div_33, _shape_param_0);  div_33 = _shape_param_0 = None
        permute_default_3: "f32[98304, 1, 9]" = torch.ops.aten.permute.default(expand_default, [0, 2, 1]);  expand_default = None
        mul_tensor: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(view_462, permute_default_3);  view_462 = permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        reshape_default: "f32[32, 512, 384, 9]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        reshape_default_1: "f32[32, 512, 3456]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_2);  reshape_default = _shape_param_2 = None
        permute_default_4: "f32[32, 3456, 512]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        reshape_default_2: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_3);  permute_default_4 = _shape_param_3 = None
        permute_default_5: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 2, 4, 3, 5]);  reshape_default_2 = None
        full_default: "f32[32, 384, 520, 1]" = torch.ops.aten.full.default([32, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(full_default, [None, None, unsqueeze_8, add_7], permute_default_5, True);  full_default = unsqueeze_8 = add_7 = permute_default_5 = None
        constant_pad_nd_default: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put_default, [0, 0, -4, -4], 0.0);  index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_dim: "f32[32, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default, -1);  constant_pad_nd_default = None
        permute_default_6: "f32[32, 512, 384]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_default: "f32[32, 512, 384]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        reshape_default_3: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        permute_default_7: "f32[768, 384]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        permute_default_8: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_tensor: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, primals_268);  convolution_23 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_9: "f32[32, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor_1: "f32[32, 512, 384]" = torch.ops.aten.mul.Tensor(view_476, permute_default_9);  view_476 = permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_4: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_5: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_default_2: "f32[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_6: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        add_tensor_1: "f32[32, 512, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, reshape_default_6);  mul_tensor_1 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default_7: "f32[16384, 384]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_8);  add_tensor_1 = _shape_param_8 = None
        permute_default_10: "f32[768, 384]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_default_11: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_10, [1, 0]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_8: "f32[16384, 384]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_9);  reshape_default_4 = _shape_param_9 = None
        permute_default_12: "f32[768, 384]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
        permute_default_13: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_12, [1, 0]);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_default_3: "f32[32, 512, 384]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_9: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_10);  clone_default_3 = _shape_param_10 = None
        permute_default_14: "f32[768, 384]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
        permute_default_15: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_14, [1, 0]);  permute_default_14 = None
        return (reshape_default_3, permute_default_8, reshape_default_7, permute_default_11, reshape_default_8, permute_default_13, reshape_default_9, permute_default_15)
