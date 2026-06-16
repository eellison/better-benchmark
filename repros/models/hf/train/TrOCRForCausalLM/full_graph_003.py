class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 256, 1024][262144, 1024, 1]cuda:0", primals_2: "f32[1024, 1024][1024, 1]cuda:0", primals_3: "f32[1024][1]cuda:0", primals_4: "f32[1024, 1024][1024, 1]cuda:0", primals_5: "f32[1024][1]cuda:0", primals_6: "f32[1024, 1024][1024, 1]cuda:0", primals_7: "f32[1024][1]cuda:0", primals_8: "f32[64, 1, 256, 256][65536, 65536, 256, 1]cuda:0", primals_9: "f32[1024, 1024][1024, 1]cuda:0", primals_10: "f32[1024][1]cuda:0", primals_11: "f32[1024][1]cuda:0", primals_12: "f32[1024][1]cuda:0", primals_13: "f32[4096, 1024][1024, 1]cuda:0", primals_14: "f32[4096][1]cuda:0", primals_15: "f32[1024, 4096][4096, 1]cuda:0", primals_16: "f32[1024][1]cuda:0", primals_17: "f32[1024][1]cuda:0", primals_18: "f32[1024][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_1: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_2: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16)
        view: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 1024]);  convert_element_type_2 = None
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [64, 256, 1024]);  addmm = None
        mul: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        convert_element_type_6: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        permute_1: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_1);  convert_element_type_6 = None
        view_3: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [64, 256, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        convert_element_type_12: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_2);  convert_element_type_12 = None
        view_5: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [64, 256, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_6: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [64, -1, 16, 64]);  view_3 = None
        permute_3: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_7: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [64, -1, 16, 64]);  view_5 = None
        permute_4: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_8: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul, [64, 256, 16, 64]);  mul = None
        permute_5: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_9: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1024, 256, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_1: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_10: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1024, 256, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_2: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_11: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1024, 256, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        bmm: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, permute_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_12: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [64, 16, 256, 256])
        add: "f32[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, primals_8);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_13: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add, [1024, 256, 256]);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_13, [-1], True)
        sub: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = None
        exp: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_20: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        bmm_1: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_20, view_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_14: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [64, 16, 256, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_7: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_4: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [64, 256, 1024]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_23: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_24: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        view_16: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [16384, 1024]);  view_15 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_3: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_23, view_16, permute_8);  convert_element_type_23 = None
        view_17: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [64, 256, 1024]);  addmm_3 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2][1]cuda:0" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:351 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 256, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt: "b8[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_1: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_17);  view_17 = None
        mul_2: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_1: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_1, mul_2);  primals_1 = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_3: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_4: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, primals_11)
        add_3: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        convert_element_type_28: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_29: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_30: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        view_18: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_30, [16384, 1024]);  convert_element_type_30 = None
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_4: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_28, view_18, permute_9);  convert_element_type_28 = None
        view_19: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [64, 256, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_5: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_6: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_4: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_4);  mul_5 = add_4 = None
        convert_element_type_35: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        convert_element_type_36: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_37: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        view_20: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [16384, 4096]);  convert_element_type_35 = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_5: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_36, view_20, permute_10);  convert_element_type_36 = None
        view_21: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [64, 256, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:378 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 256, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_1: "b8[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_8: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_21);  view_21 = None
        mul_9: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_5: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3, mul_9);  add_3 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_10: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_17)
        add_7: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_18);  mul_11 = primals_18 = None
        div_1: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_11: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_15: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        div_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        permute_19: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        permute_24: "bf16[1024, 256, 256][65536, 1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_20, [0, 2, 1]);  convert_element_type_20 = None
        permute_25: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_26: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_27: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_32: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_36: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        permute_40: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add_7, primals_8, primals_11, primals_17, view, bmm, amax, sum_1, view_16, gt, mul_3, view_18, addmm_4, view_20, gt_1, mul_10, div_1, permute_11, permute_15, div_2, permute_19, permute_24, permute_25, permute_26, permute_27, permute_32, permute_36, permute_40)
