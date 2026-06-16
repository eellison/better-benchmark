class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024][1]cuda:0", primals_2: "f32[1024][1]cuda:0", primals_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0", primals_4: "f32[1024, 1024][1024, 1]cuda:0", primals_5: "f32[1024][1]cuda:0", primals_6: "f32[1024, 1024][1024, 1]cuda:0", primals_7: "f32[1024][1]cuda:0", primals_8: "f32[1024, 1024][1024, 1]cuda:0", primals_9: "f32[1024][1]cuda:0", primals_10: "f32[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0", primals_11: "f32[1024, 1024][1024, 1]cuda:0", primals_12: "f32[1024][1]cuda:0", primals_13: "f32[1024][1]cuda:0", primals_14: "f32[1024][1]cuda:0", primals_15: "f32[4096, 1024][1024, 1]cuda:0", primals_16: "f32[4096][1]cuda:0", primals_17: "f32[1024, 4096][4096, 1]cuda:0", primals_18: "f32[1024][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_1: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_2: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [4096, 1024]);  convert_element_type_2 = None
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 128, 1024]);  addmm = None
        mul_2: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        convert_element_type_6: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_1: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_1);  convert_element_type_6 = None
        view_3: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        convert_element_type_12: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_2);  convert_element_type_12 = None
        view_5: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 128, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_6: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [32, 128, -1, 64]);  view_3 = None
        permute_3: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_7: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 128, -1, 64]);  view_5 = None
        permute_4: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_8: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_2, [32, 128, 16, 64]);  mul_2 = None
        permute_5: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_9: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [512, 128, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_1: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_10: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [512, 128, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_2: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_11: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [512, 128, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, permute_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_12: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128])
        add_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, primals_10);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_2, full_default);  add_2 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_13: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum, [512, 128, 128]);  maximum = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_13, [-1], True)
        sub_1: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = None
        exp: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[3][1]cuda:0" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:226 in forward, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_2: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_3: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, div);  div = None
        mul_4: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_20: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        bmm_1: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_20, view_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_16: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 16, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_7: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_3: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [32, 128, 1024]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_23: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_24: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        view_18: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [4096, 1024]);  view_17 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_23, view_18, permute_8);  convert_element_type_23 = None
        view_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:311 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_1: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_1: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_5: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_19);  view_19 = None
        mul_6: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_6);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_3)
        mul_7: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_8: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, primals_13);  mul_7 = None
        add_5: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, primals_14);  mul_8 = primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        convert_element_type_28: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_29: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_30: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_20: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_30, [4096, 1024]);  convert_element_type_30 = None
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_4: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_28, view_20, permute_9);  convert_element_type_28 = None
        view_21: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_9: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_10: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_6: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, add_6);  mul_9 = add_6 = None
        convert_element_type_35: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        convert_element_type_36: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_37: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        view_22: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [4096, 4096]);  convert_element_type_35 = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_5: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_36, view_22, permute_10);  convert_element_type_36 = None
        view_23: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 128, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:335 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_2: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_12: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_23);  view_23 = None
        mul_13: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_7: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3, mul_13);  add_3 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_11: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_15: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        permute_19: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        permute_24: "bf16[512, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_20, [0, 2, 1]);  convert_element_type_20 = None
        permute_25: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_26: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_27: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        permute_32: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        permute_36: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        permute_40: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add_7, primals_1, primals_3, primals_10, primals_13, getitem_1, rsqrt, view, bmm, amax, sum_1, gt, view_18, addmm_3, gt_1, getitem_3, rsqrt_1, view_20, addmm_4, view_22, gt_2, permute_11, permute_15, permute_19, permute_24, permute_25, permute_26, permute_27, permute_32, permute_36, permute_40)
