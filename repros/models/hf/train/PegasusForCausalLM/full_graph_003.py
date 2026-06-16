class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024][1]cuda:0", primals_2: "f32[1024][1]cuda:0", primals_3: "f32[128, 128, 1024][131072, 1024, 1]cuda:0", primals_4: "f32[1024, 1024][1024, 1]cuda:0", primals_5: "f32[1024][1]cuda:0", primals_6: "f32[1024, 1024][1024, 1]cuda:0", primals_7: "f32[1024][1]cuda:0", primals_8: "f32[1024, 1024][1024, 1]cuda:0", primals_9: "f32[1024][1]cuda:0", primals_10: "b8[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0", primals_11: "f32[1024, 1024][1024, 1]cuda:0", primals_12: "f32[1024][1]cuda:0", primals_13: "f32[1024][1]cuda:0", primals_14: "f32[1024][1]cuda:0", primals_15: "f32[4096, 1024][1024, 1]cuda:0", primals_16: "f32[4096][1]cuda:0", primals_17: "f32[1024, 4096][4096, 1]cuda:0", primals_18: "f32[1024][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_1: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_2: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 1024]);  convert_element_type_2 = None
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [128, 128, 1024]);  addmm = None
        view_2: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [128, 128, -1, 64]);  view_1 = None
        permute_1: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        convert_element_type_6: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_2);  convert_element_type_6 = None
        view_4: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 128, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        convert_element_type_12: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_3: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_3);  convert_element_type_12 = None
        view_6: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 128, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [128, 128, -1, 64]);  view_4 = None
        permute_4: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [128, 128, -1, 64]);  view_6 = None
        permute_5: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, True, scale = 0.125)
        getitem_2: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0]
        getitem_3: "f32[128, 16, 128, 1][2048, 128, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention[1]
        getitem_8: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[6]
        getitem_9: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [128, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_18: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_19: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        view_10: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [16384, 1024]);  view_9 = None
        permute_7: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_3: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_18, view_10, permute_7);  convert_element_type_18 = view_10 = None
        view_11: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 128, 1024])

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2][1]cuda:0" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:376 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 128, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt: "b8[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_2: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_11: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_12)
        mul_4: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_5: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, primals_13);  mul_4 = None
        add_4: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, primals_14);  mul_5 = primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        convert_element_type_23: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_24: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_25: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        view_12: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 1024]);  convert_element_type_25 = None
        permute_8: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_4: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_23, view_12, permute_8);  convert_element_type_23 = None
        view_13: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [128, 128, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_29: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_6: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, 0.5)
        mul_7: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, 0.7071067811865476);  convert_element_type_29 = None
        erf: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_5: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_5);  mul_6 = add_5 = None
        convert_element_type_30: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        convert_element_type_31: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_32: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        view_14: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_30, [16384, 4096]);  convert_element_type_30 = None
        permute_9: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        addmm_5: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_31, view_14, permute_9);  convert_element_type_31 = None
        view_15: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 128, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:400 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 128, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_1: "b8[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_9: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_15);  view_15 = None
        mul_10: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_6: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, mul_10);  add_2 = mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_10: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_14: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        permute_18: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        permute_25: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        permute_29: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_34: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add_6, primals_1, primals_3, primals_13, getitem_1, rsqrt, view, permute_1, permute_4, permute_5, where, getitem_2, getitem_3, getitem_8, getitem_9, addmm_3, gt, getitem_12, rsqrt_1, view_12, addmm_4, view_14, gt_1, permute_10, permute_14, permute_18, permute_25, permute_29, permute_34)
