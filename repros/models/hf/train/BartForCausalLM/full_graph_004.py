class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 1024, 1024]", primals_2: "f32[1024, 1024]", primals_3: "f32[1024]", primals_4: "f32[1024, 1024]", primals_5: "f32[1024]", primals_6: "f32[1024, 1024]", primals_7: "f32[1024]", primals_8: "b8[8, 1, 1024, 1024]", primals_9: "f32[1024, 1024]", primals_10: "f32[1024]", primals_11: "f32[1024]", primals_12: "f32[1024]", primals_13: "f32[4096, 1024]", primals_14: "f32[4096]", primals_15: "f32[1024, 4096]", primals_16: "f32[1024]", primals_17: "f32[1024]", primals_18: "f32[1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[8192, 1024]" = torch.ops.aten.reshape.default(primals_1, [8192, 1024])
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_3, view, permute);  primals_3 = permute = None
        view_1: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm, [8, 1024, 1024]);  addmm = None
        view_2: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_1, [8, 1024, -1, 64]);  view_1 = None
        permute_1: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_5, view, permute_2);  primals_5 = permute_2 = None
        view_4: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_2: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_7, view, permute_3);  primals_7 = view = permute_3 = None
        view_6: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_4, [8, 1024, -1, 64]);  view_4 = None
        permute_4: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_6, [8, 1024, -1, 64]);  view_6 = None
        permute_5: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(primals_8, full_default_1, full_default);  primals_8 = full_default_1 = full_default = None
        expand: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where, [8, 16, 1024, 1024])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_4, permute_5, expand, True, scale = 0.125);  expand = None
        getitem: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_1: "f32[8, 16, 1024]" = _scaled_dot_product_efficient_attention[1]
        getitem_2: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_3: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_6, [8, 1024, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_3: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_10, view_10, permute_7);  primals_10 = view_10 = permute_7 = None
        view_11: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 1024])

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:362 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(primals_1, mul_1);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 1024, 1]" = var_mean[0]
        getitem_5: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add, getitem_5);  add = None
        mul_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_2, primals_11);  mul_2 = None
        add_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_2, [8192, 1024])
        permute_8: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_4: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_14, view_12, permute_8);  primals_14 = permute_8 = None
        view_13: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_4: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_3: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_4, add_3);  mul_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_6, [8192, 4096]);  mul_6 = None
        permute_9: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_5: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_16, view_14, permute_9);  primals_16 = permute_9 = None
        view_15: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:386 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 1024, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_7: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_1, view_15);  view_15 = None
        mul_8: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_4: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_8);  add_2 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_7: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_9: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_10: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_9, primals_17)
        add_6: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_10, primals_18);  mul_10 = primals_18 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        return (add_6, primals_1, primals_2, primals_4, primals_6, primals_9, primals_11, primals_13, primals_15, primals_17, permute_1, permute_4, permute_5, where, getitem, getitem_1, getitem_2, getitem_3, addmm_3, gt, getitem_5, rsqrt, view_12, addmm_4, view_14, gt_1, mul_9, div)
