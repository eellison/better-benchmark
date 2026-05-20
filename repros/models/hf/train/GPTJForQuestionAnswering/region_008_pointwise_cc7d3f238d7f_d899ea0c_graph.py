class GraphModule(torch.nn.Module):
    def forward(self, mm_438: "f32[128, 16384]", addmm: "f32[128, 16384]", primals_10: "f32[16384, 4096]", getitem_309: "f32[1, 16, 128, 256]", getitem_310: "f32[1, 16, 128, 256]", unsqueeze_12: "f32[1, 128, 1, 32, 1]", full_default_13: "f32[1, 128, 16, 64]", unsqueeze_14: "f32[1, 128, 1, 32, 1]", full_default_17: "f32[1, 128, 16, 256]", getitem_311: "f32[1, 16, 128, 256]", primals_7: "f32[4096, 4096]", primals_6: "f32[4096, 4096]", primals_5: "f32[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        reshape_default: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_438, _shape_param_0);  mm_438 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        reshape_default_1: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_1: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_2: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None
        mul_tensor_5: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        reshape_default_2: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_1: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_default_2: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_309, [0, 2, 1, 3]);  getitem_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_default_3: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_310, [0, 2, 1, 3]);  getitem_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 64)
        slice_tensor_1: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 64, 256);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_default_3, 3, 0, 64)
        slice_tensor_3: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_default_3, 3, 64, 256);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_12, _shape_param_3);  unsqueeze_12 = _shape_param_3 = None
        clone_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_3: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_tensor_11: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, reshape_default_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        reshape_default_4: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_tensor_11, _shape_param_5);  mul_tensor_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_int: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(reshape_default_4, -1, 0)
        select_int_1: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(reshape_default_4, -1, 1);  reshape_default_4 = None
        neg_default: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_int);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_default: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_13, neg_default, 3, 1, 9223372036854775807, 2);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_default_1: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_13, select_int_1, 3, 0, 9223372036854775807, 2);  select_int_1 = None
        add_tensor_4: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_14, _shape_param_6);  unsqueeze_14 = _shape_param_6 = None
        clone_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_tensor_12: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, reshape_default_5);  slice_tensor = None
        add_tensor_5: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_12);  add_tensor_4 = mul_tensor_12 = None
        mul_tensor_13: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, reshape_default_3);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        reshape_default_6: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_tensor_13, _shape_param_8);  mul_tensor_13 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_int_2: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(reshape_default_6, -1, 0)
        select_int_3: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(reshape_default_6, -1, 1);  reshape_default_6 = None
        neg_default_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_int_2);  select_int_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_default_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_13, neg_default_1, 3, 1, 9223372036854775807, 2);  neg_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_default_3: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_13, select_int_3, 3, 0, 9223372036854775807, 2);  full_default_13 = select_int_3 = None
        add_tensor_6: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_tensor_14: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, reshape_default_5);  slice_tensor_2 = reshape_default_5 = None
        add_tensor_7: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_14);  add_tensor_6 = mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_default_4: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_17, slice_tensor_1, 3, 64, 9223372036854775807);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_default_5: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_17, add_tensor_5, 3, 0, 64);  add_tensor_5 = None
        add_tensor_8: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_default_4, slice_scatter_default_5);  slice_scatter_default_4 = slice_scatter_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_default_6: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_17, slice_tensor_3, 3, 64, 9223372036854775807);  slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_default_7: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_17, add_tensor_7, 3, 0, 64);  full_default_17 = add_tensor_7 = None
        add_tensor_9: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_default_6, slice_scatter_default_7);  slice_scatter_default_6 = slice_scatter_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_4: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_311, [0, 2, 1, 3]);  getitem_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_default_2: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_7: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_9);  clone_default_2 = _shape_param_9 = None
        reshape_default_8: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_9, _shape_param_10);  add_tensor_9 = _shape_param_10 = None
        reshape_default_9: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_8, _shape_param_11);  add_tensor_8 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_10: "f32[128, 4096]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_12);  reshape_default_7 = _shape_param_12 = None
        permute_default_5: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_6: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_11: "f32[128, 4096]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_13);  reshape_default_8 = _shape_param_13 = None
        permute_default_7: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        reshape_default_12: "f32[128, 4096]" = torch.ops.aten.reshape.default(reshape_default_9, _shape_param_14);  reshape_default_9 = _shape_param_14 = None
        permute_default_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_10: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None
        return (reshape_default_2, permute_default_1, reshape_default_10, permute_default_6, reshape_default_11, permute_default_8, reshape_default_12, permute_default_10)
