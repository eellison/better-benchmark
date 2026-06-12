class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[768][1]cuda:0", primals_2: "f32[768][1]cuda:0", primals_3: "f32[4, 2048, 768][1572864, 768, 1]cuda:0", primals_4: "f32[768, 768][768, 1]cuda:0", primals_5: "f32[768][1]cuda:0", primals_6: "f32[768, 768][768, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768, 768][768, 1]cuda:0", primals_9: "f32[768][1]cuda:0", primals_10: "b8[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_11: "f32[768, 768][768, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[3072, 768][768, 1]cuda:0", primals_16: "f32[3072][1]cuda:0", primals_17: "f32[768, 3072][3072, 1]cuda:0", primals_18: "f32[768][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_1: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_2: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [8192, 768]);  convert_element_type_2 = None
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [4, 2048, 768]);  addmm = None
        mul_2: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_2: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_2, [4, -1, 12, 64]);  mul_2 = None
        permute_1: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        convert_element_type_6: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_7: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_2);  convert_element_type_6 = None
        view_4: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [4, 2048, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        convert_element_type_12: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_3: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_3);  convert_element_type_12 = None
        view_6: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [4, 2048, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_7: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [4, -1, 12, 64]);  view_4 = None
        permute_4: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_8: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [4, -1, 12, 64]);  view_6 = None
        permute_5: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, True, scale = 1.0)
        getitem_2: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0]
        getitem_3: "f32[4, 12, 2048, 1][24576, 2048, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention[1]
        getitem_8: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[6]
        getitem_9: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_9: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [4, 2048, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_18: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_19: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        view_10: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [8192, 768]);  view_9 = None
        permute_7: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_3: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_18, view_10, permute_7);  convert_element_type_18 = view_10 = None
        view_11: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [4, 2048, 768])

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2][1]cuda:0" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([4, 2048, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt: "b8[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_3: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_4: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_4);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_12: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(view_12, [1], correction = 0, keepdim = True)
        getitem_11: "f32[8192, 1][1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[8192, 1][1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_12, getitem_12)
        mul_5: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_6: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, primals_13);  mul_5 = None
        add_4: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, primals_14);  mul_6 = primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_23: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_24: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        permute_8: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_4: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_23, convert_element_type_25, permute_8);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        convert_element_type_29: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_30: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        permute_9: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_5: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_29, relu, permute_9);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:245 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8192, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_1: "b8[8192, 768][768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, addmm_5);  addmm_5 = None
        mul_8: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_5: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, mul_8);  view_12 = mul_8 = None
        view_13: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [4, 2048, 768]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_10: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_14: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        permute_18: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        permute_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        permute_29: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        permute_34: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (view_13, primals_1, primals_3, primals_13, getitem_1, rsqrt, view, permute_1, permute_4, permute_5, where, getitem_2, getitem_3, getitem_8, getitem_9, addmm_3, gt, getitem_12, rsqrt_1, convert_element_type_25, relu, gt_1, permute_10, permute_14, permute_18, permute_25, permute_29, permute_34)
