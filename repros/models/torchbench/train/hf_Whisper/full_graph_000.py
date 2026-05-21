class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[384]", primals_2: "f32[384]", primals_3: "f32[8, 1500, 384]", primals_4: "f32[384, 384]", primals_5: "f32[384]", primals_6: "f32[384, 384]", primals_7: "f32[384, 384]", primals_8: "f32[384]", primals_9: "f32[384, 384]", primals_10: "f32[384]", primals_11: "f32[384]", primals_12: "f32[384]", primals_13: "f32[1536, 384]", primals_14: "f32[1536]", primals_15: "f32[384, 1536]", primals_16: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(primals_3, memory_format = torch.contiguous_format)
        var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean[1];  var_mean = None
        add: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
        mul: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view: "f32[12000, 384]" = torch.ops.aten.reshape.default(add_1, [12000, 384]);  add_1 = None
        permute: "f32[384, 384]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm: "f32[12000, 384]" = torch.ops.aten.addmm.default(primals_5, view, permute);  primals_5 = permute = None
        view_1: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm, [8, 1500, 384]);  addmm = None
        mul_2: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        view_2: "f32[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(mul_2, [8, 1500, -1, 64]);  mul_2 = None
        permute_1: "f32[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_1: "f32[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        permute_2: "f32[384, 384]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm: "f32[12000, 384]" = torch.ops.aten.mm.default(view, permute_2);  permute_2 = None
        view_4: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm, [8, 1500, 384]);  mm = None
        view_5: "f32[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_4, [8, -1, 6, 64]);  view_4 = None
        permute_3: "f32[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_2: "f32[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        permute_4: "f32[384, 384]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm_1: "f32[12000, 384]" = torch.ops.aten.addmm.default(primals_8, view, permute_4);  primals_8 = permute_4 = None
        view_7: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_1, [8, 1500, 384]);  addmm_1 = None
        view_8: "f32[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_7, [8, -1, 6, 64]);  view_7 = None
        permute_5: "f32[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_3: "f32[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_1, clone_2, clone_3, None, True, scale = 1.0)
        getitem_2: "f32[8, 6, 1500, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_3: "f32[8, 6, 1504]" = _scaled_dot_product_efficient_attention[1]
        getitem_4: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_5: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(permute_6, [8, 1500, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[12000, 384]" = torch.ops.aten.reshape.default(view_9, [12000, 384]);  view_9 = None
        permute_7: "f32[384, 384]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_2: "f32[12000, 384]" = torch.ops.aten.addmm.default(primals_10, view_10, permute_7);  primals_10 = view_10 = permute_7 = None
        view_11: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_2, [8, 1500, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(primals_3, view_11);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_5: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format)
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 1500, 1]" = var_mean_1[0]
        getitem_7: "f32[8, 1500, 1]" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(clone_5, getitem_7);  clone_5 = None
        mul_3: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_4: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_3, primals_11);  mul_3 = None
        add_4: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "f32[12000, 384]" = torch.ops.aten.reshape.default(add_4, [12000, 384]);  add_4 = None
        permute_8: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_3: "f32[12000, 1536]" = torch.ops.aten.addmm.default(primals_14, view_12, permute_8);  primals_14 = permute_8 = None
        view_13: "f32[8, 1500, 1536]" = torch.ops.aten.reshape.default(addmm_3, [8, 1500, 1536])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_5: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_6: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[8, 1500, 1536]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_5: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "f32[12000, 1536]" = torch.ops.aten.reshape.default(mul_7, [12000, 1536]);  mul_7 = None
        permute_9: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_4: "f32[12000, 384]" = torch.ops.aten.addmm.default(primals_16, view_14, permute_9);  primals_16 = permute_9 = None
        view_15: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_4, [8, 1500, 384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_6: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_2, view_15);  add_2 = view_15 = None
        return (add_6, primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_11, primals_13, primals_15, getitem_1, rsqrt, view, clone_1, clone_2, clone_3, getitem_2, getitem_3, getitem_4, getitem_5, addmm_2, getitem_7, rsqrt_1, view_12, addmm_3, view_14)
