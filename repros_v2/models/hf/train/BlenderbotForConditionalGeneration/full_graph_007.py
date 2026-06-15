class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2560][1]cuda:0", primals_2: "f32[2560][1]cuda:0", primals_3: "f32[16, 128, 2560][327680, 2560, 1]cuda:0", primals_4: "f32[2560, 2560][2560, 1]cuda:0", primals_5: "f32[2560][1]cuda:0", primals_6: "f32[2560, 2560][2560, 1]cuda:0", primals_7: "f32[2560][1]cuda:0", primals_8: "f32[2560, 2560][2560, 1]cuda:0", primals_9: "f32[2560][1]cuda:0", primals_10: "b8[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0", primals_11: "f32[2560, 2560][2560, 1]cuda:0", primals_12: "f32[2560][1]cuda:0", primals_13: "f32[16, 128, 2560][327680, 2560, 1]cuda:0", primals_14: "f32[2560][1]cuda:0", primals_15: "f32[2560][1]cuda:0", primals_16: "f32[2560, 2560][2560, 1]cuda:0", primals_17: "f32[2560][1]cuda:0", primals_18: "f32[2560, 2560][2560, 1]cuda:0", primals_19: "f32[2560][1]cuda:0", primals_20: "f32[2560, 2560][2560, 1]cuda:0", primals_21: "f32[2560][1]cuda:0", primals_22: "b8[16, 1, 128, 128][0, 128, 1, 0]cuda:0", primals_23: "f32[2560, 2560][2560, 1]cuda:0", primals_24: "f32[2560][1]cuda:0", primals_25: "f32[2560][1]cuda:0", primals_26: "f32[2560][1]cuda:0", primals_27: "f32[10240, 2560][2560, 1]cuda:0", primals_28: "f32[10240][1]cuda:0", primals_29: "f32[2560, 10240][10240, 1]cuda:0", primals_30: "f32[2560][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_1: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_2: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [2048, 2560]);  convert_element_type_2 = None
        permute: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [16, 128, 2560]);  addmm = None
        view_2: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [16, 128, -1, 80]);  view_1 = None
        permute_1: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        convert_element_type_6: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_7: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_2: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_2);  convert_element_type_6 = None
        view_4: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [16, 128, 2560]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        convert_element_type_12: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_13: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_3: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_3);  convert_element_type_12 = None
        view_6: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [16, 128, 2560]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [16, 128, -1, 80]);  view_4 = None
        permute_4: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [16, 128, -1, 80]);  view_6 = None
        permute_5: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, True, scale = 0.11180339887498948)
        getitem_2: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0]
        getitem_3: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention[1]
        getitem_8: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[6]
        getitem_9: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [16, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_18: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_19: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        view_10: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [2048, 2560]);  view_9 = None
        permute_7: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_3: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_18, view_10, permute_7);  convert_element_type_18 = view_10 = None
        view_11: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560])

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[3][1]cuda:0" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_2: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 2560], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_2: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt: "b8[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_2: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_12)
        mul_4: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_5: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, primals_14);  mul_4 = None
        add_4: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, primals_15);  mul_5 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_23: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_24: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_25: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        view_12: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [2048, 2560]);  convert_element_type_25 = None
        permute_8: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_4: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_23, view_12, permute_8);  convert_element_type_23 = None
        view_13: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 2560]);  addmm_4 = None
        view_14: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_13, [16, 128, -1, 80]);  view_13 = None
        permute_9: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        convert_element_type_29: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_30: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_31: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        view_15: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [2048, 2560]);  convert_element_type_31 = None
        permute_10: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_5: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_29, view_15, permute_10);  convert_element_type_29 = None
        view_16: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [16, 128, 2560]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        convert_element_type_35: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_36: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_11: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        addmm_6: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_35, view_15, permute_11);  convert_element_type_35 = None
        view_18: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [16, 128, 2560]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_19: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [16, 128, -1, 80]);  view_16 = None
        permute_12: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_20: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_18, [16, 128, -1, 80]);  view_18 = None
        permute_13: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        where_1: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(primals_22, full_default_1, full_default);  primals_22 = full_default_1 = full_default = None
        mul_6: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_9, 0.334370152488211);  permute_9 = None
        permute_14: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_12, [0, 1, 3, 2]);  permute_12 = None
        mul_7: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_14, 0.334370152488211);  permute_14 = None
        expand: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_6, [16, 32, 128, 80]);  mul_6 = None
        clone: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_21: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [512, 128, 80]);  clone = None
        expand_1: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_7, [16, 32, 80, 128]);  mul_7 = None
        clone_1: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_22: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [512, 80, 128]);  clone_1 = None
        bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_21, view_22)
        view_23: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [16, 32, 128, 128]);  bmm = None
        add_5: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_23, where_1);  view_23 = where_1 = None
        convert_element_type_43: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        amax: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_2: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax);  convert_element_type_43 = amax = None
        exp: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_44: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_5, -inf);  add_5 = None
        logical_not: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_4: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_4, convert_element_type_44);  logical_not_1 = full_default_4 = convert_element_type_44 = None
        expand_2: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_2, [16, 32, 128, 128])
        view_24: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [512, 128, 128]);  expand_2 = None
        expand_3: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_13, [16, 32, 128, 80]);  permute_13 = None
        clone_2: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_25: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [512, 128, 80]);  clone_2 = None
        bmm_1: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = None
        view_26: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [16, 32, 128, 80]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_3: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_27: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [16, 128, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_47: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_48: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        view_28: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [2048, 2560]);  view_27 = None
        permute_16: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        addmm_7: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_47, view_28, permute_16);  convert_element_type_47 = None
        view_29: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [16, 128, 2560]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:382 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 2560], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_1: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_1: "b8[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_8: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_29);  view_29 = None
        mul_9: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_6: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, mul_9);  add_2 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_6, getitem_14);  getitem_14 = None
        mul_10: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_11: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_25)
        add_8: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_26);  mul_11 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        convert_element_type_52: "bf16[10240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_53: "bf16[10240, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_54: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        view_30: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_54, [2048, 2560]);  convert_element_type_54 = None
        permute_17: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        addmm_8: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_52, view_30, permute_17);  convert_element_type_52 = None
        view_31: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 128, 10240])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_58: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        mul_12: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.5)
        mul_13: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476);  convert_element_type_58 = None
        erf: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_9);  mul_12 = add_9 = None
        convert_element_type_59: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        convert_element_type_60: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_61: "bf16[2560, 10240][10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        view_32: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_59, [2048, 10240]);  convert_element_type_59 = None
        permute_18: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        addmm_9: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_60, view_32, permute_18);  convert_element_type_60 = None
        view_33: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [16, 128, 2560]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:391 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 2560], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_2: "b8[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_15: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_33);  view_33 = None
        mul_16: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_10: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, mul_16);  add_6 = mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_19: "bf16[2560, 10240][10240, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_23: "bf16[10240, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        div_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 2560);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        permute_27: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_33: "bf16[512, 80, 128][10240, 1, 80]cuda:0" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_34: "bf16[512, 80, 128][10240, 1, 80]cuda:0" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
        permute_35: "bf16[512, 128, 80][10240, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_39: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_43: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_48: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        permute_52: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_59: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_63: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_68: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add_10, primals_1, primals_3, primals_14, primals_25, getitem_1, rsqrt, view, permute_1, permute_4, permute_5, where, getitem_2, getitem_3, getitem_8, getitem_9, addmm_3, gt, getitem_12, rsqrt_1, view_12, view_15, where_2, view_28, gt_1, mul_10, view_30, addmm_8, view_32, gt_2, permute_19, permute_23, div_1, permute_27, permute_33, permute_34, permute_35, permute_39, permute_43, permute_48, permute_52, permute_59, permute_63, permute_68)
