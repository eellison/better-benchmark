class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[256]", arg1_1: "f32[256]", arg2_1: "f32[8, 4096, 256]", arg3_1: "f32[768, 256]", arg4_1: "f32[768, 256]", arg5_1: "f32[768, 256]", arg6_1: "f32[256, 768]", arg7_1: "f32[8, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 4096, 1]" = var_mean[0]
        getitem_1: "f32[8, 4096, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        _tensor_constant0: "i64[]" = self._tensor_constant0
        lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        convert_element_type: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy, torch.float64);  lift_fresh_copy = convert_element_type = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1332 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1265 in forward, code: attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.prims.inductor_random.default([8, 12, 64, 64, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 12, 64, 64, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.05);  inductor_random_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub: "f32[8, 4096, 256]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  arg2_1 = getitem_1 = None
        add: "f32[8, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 4096, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_1: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1162 in forward, code: query_vectors = self.query(hidden_states)
        view: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, [32768, 256])
        permute: "f32[256, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f32[32768, 768]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm, [8, 4096, 768]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_6: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_1, [8, 4096, 12, 64]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_3: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_9: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_3, [8, 12, 64, 64, 64]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_9, [8, 12, 64, 64, 64]);  view_9 = None
        clone: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_14: "f32[6144, 64, 64]" = torch.ops.aten.reshape.default(clone, [6144, 64, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        view_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, [32768, 256])
        permute_1: "f32[256, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f32[32768, 768]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm_1, [8, 4096, 768]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_7: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_3, [8, 4096, 12, 64]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_4: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "f32[8, 12, 4096, 64]" = torch.ops.aten.div.Tensor(permute_4, full_default);  permute_4 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_10: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(div, [8, 12, 64, 64, 64]);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_1: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_10, 2, -1, 9223372036854775807)
        slice_2: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_10, 2, 0, -1)
        cat: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_1, slice_2], 2);  slice_1 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_1: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat, view_10], 3);  cat = view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_6: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_1, [0, 1, 2, 4, 3]);  cat_1 = None
        expand_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_6, [8, 12, 64, 64, 128]);  permute_6 = None
        view_15: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_1, [6144, 64, 128]);  expand_1 = None
        bmm: "f32[6144, 64, 128]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
        view_16: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm, [8, 12, 64, 64, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1258 in forward, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        amax: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.amax.default(view_16, [-1], True)
        abs_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax)
        eq: "b8[8, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq, full_default_1, amax);  eq = full_default_1 = amax = None
        sub_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_16, where)
        exp: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        log: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        add_2: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log, where);  log = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1259 in forward, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_16, add_2);  view_16 = add_2 = None
        exp_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1265 in forward, code: attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)
        mul_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.mul.Tensor(gt, exp_1);  gt = exp_1 = None
        mul_3: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.mul.Tensor(mul_2, 1.0526315789473684);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(mul_3, [8, 12, 64, 64, 128]);  mul_3 = None
        view_17: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_2, [6144, 64, 128]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1164 in forward, code: value_vectors = self.value(hidden_states)
        view_4: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, [32768, 256]);  add_1 = None
        permute_2: "f32[256, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "f32[32768, 768]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 4096, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_8: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_5, [8, 4096, 12, 64]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_5: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_11: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_5, [8, 12, 64, 64, 64]);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_3: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_11, 2, -1, 9223372036854775807)
        slice_4: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_11, 2, 0, -1)
        cat_2: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_3, slice_4], 2);  slice_3 = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_3: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_2, view_11], 3);  cat_2 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_3: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_3, [8, 12, 64, 128, 64]);  cat_3 = None
        view_18: "f32[6144, 128, 64]" = torch.ops.aten.reshape.default(expand_3, [6144, 128, 64]);  expand_3 = None
        bmm_1: "f32[6144, 64, 64]" = torch.ops.aten.bmm.default(view_17, view_18);  view_17 = view_18 = None
        view_19: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 64, 64, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1275 in forward, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_20: "f32[8, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_19, [8, 12, 4096, 64]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_7: "f32[8, 4096, 12, 64]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_1: "f32[8, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_21: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(clone_1, [8, 4096, 768]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_22: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_21, [32768, 768]);  view_21 = None
        permute_8: "f32[768, 256]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_3: "f32[32768, 256]" = torch.ops.aten.mm.default(view_22, permute_8);  view_22 = permute_8 = None
        view_23: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(mm_3, [8, 4096, 256]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1332 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_4: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt_1, view_23);  gt_1 = view_23 = None
        mul_5: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_4, 1.0526315789473684);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in torch_dynamo_resume_in_forward_at_1541, code: attn_output = prev_attn_output + attn_output
        add_3: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(arg7_1, mul_5);  arg7_1 = None
        return (mul_5, add_3)
