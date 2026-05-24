import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 4096, 256]", arg1_1: "f16[256]", arg2_1: "f16[256]", arg3_1: "f16[768, 256]", arg4_1: "f16[768, 256]", arg5_1: "f16[768, 256]", arg6_1: "f16[256, 768]", arg7_1: "f16[256]", arg8_1: "f16[256]", arg9_1: "f16[512, 256]", arg10_1: "f16[512]", arg11_1: "f16[256, 512]", arg12_1: "f16[256]", arg13_1: "f16[256]", arg14_1: "f16[256]", arg15_1: "f16[768, 256]", arg16_1: "f16[768, 256]", arg17_1: "f16[]", arg18_1: "f16[]", arg19_1: "f16[256, 768]", arg20_1: "f16[256]", arg21_1: "f16[256]", arg22_1: "f16[512, 256]", arg23_1: "f16[512]", arg24_1: "f16[256, 512]", arg25_1: "f16[256]", arg26_1: "f16[256]", arg27_1: "f16[256]", arg28_1: "f16[768, 256]", arg29_1: "f16[768, 256]", arg30_1: "f16[768, 256]", arg31_1: "f16[256, 768]", arg32_1: "f16[256]", arg33_1: "f16[256]", arg34_1: "f16[512, 256]", arg35_1: "f16[512]", arg36_1: "f16[256, 512]", arg37_1: "f16[256]", arg38_1: "f16[256]", arg39_1: "f16[256]", arg40_1: "f16[768, 256]", arg41_1: "f16[768, 256]", arg42_1: "f16[]", arg43_1: "f16[]", arg44_1: "f16[256, 768]", arg45_1: "f16[256]", arg46_1: "f16[256]", arg47_1: "f16[512, 256]", arg48_1: "f16[512]", arg49_1: "f16[256, 512]", arg50_1: "f16[256]", arg51_1: "f16[256]", arg52_1: "f16[256]", arg53_1: "f16[768, 256]", arg54_1: "f16[768, 256]", arg55_1: "f16[768, 256]", arg56_1: "f16[256, 768]", arg57_1: "f16[256]", arg58_1: "f16[256]", arg59_1: "f16[512, 256]", arg60_1: "f16[512]", arg61_1: "f16[256, 512]", arg62_1: "f16[256]", arg63_1: "f16[256]", arg64_1: "f16[256]", arg65_1: "f16[768, 256]", arg66_1: "f16[768, 256]", arg67_1: "f16[]", arg68_1: "f16[]", arg69_1: "f16[256, 768]", arg70_1: "f16[256]", arg71_1: "f16[256]", arg72_1: "f16[512, 256]", arg73_1: "f16[512]", arg74_1: "f16[256, 512]", arg75_1: "f16[256]", arg76_1: "f16[512]", arg77_1: "f16[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1780 in forward, code: hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        unsqueeze: "f16[1, 4096, 1, 256]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand: "f16[1, 4096, 2, 256]" = torch.ops.aten.expand.default(unsqueeze, [1, 4096, 2, 256]);  unsqueeze = None
        clone: "f16[1, 4096, 2, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(clone, [1, 4096, 512]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1668 in forward, code: hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)
        split = torch.ops.aten.split.Tensor(view, 256, -1);  view = None
        getitem: "f16[1, 4096, 256]" = split[0]
        getitem_1: "f16[1, 4096, 256]" = split[1];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        clone_2: "f16[1, 4096, 256]" = torch.ops.aten.clone.default(getitem, memory_format = torch.contiguous_format)
        convert_element_type: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(clone_2, torch.float32);  clone_2 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 4096, 1]" = var_mean[0]
        getitem_3: "f32[1, 4096, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        _tensor_constant0: "i64[]" = self._tensor_constant0
        lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        convert_element_type_8: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy, torch.float64);  lift_fresh_copy = convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_3);  convert_element_type = getitem_3 = None
        add: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type_1: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1162 in forward, code: query_vectors = self.query(hidden_states)
        view_1: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 256])
        permute: "f16[256, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f16[4096, 768]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm, [1, 4096, 768]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_7: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_2, [1, 4096, 12, 64]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_3: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_10: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_3, [1, 12, 64, 64, 64]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_1: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_10, [1, 12, 64, 64, 64]);  view_10 = None
        clone_3: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_15: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(clone_3, [768, 64, 64]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        view_3: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 256])
        permute_1: "f16[256, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f16[4096, 768]" = torch.ops.aten.mm.default(view_3, permute_1);  view_3 = permute_1 = None
        view_4: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_1, [1, 4096, 768]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_8: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, 4096, 12, 64]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_4: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "f16[1, 12, 4096, 64]" = torch.ops.aten.div.Tensor(permute_4, full_default);  permute_4 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_11: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(div, [1, 12, 64, 64, 64]);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_1: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_11, 2, -1, 9223372036854775807)
        slice_2: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_11, 2, 0, -1)
        cat: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_1, slice_2], 2);  slice_1 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_1: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat, view_11], 3);  cat = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_6: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_1, [0, 1, 2, 4, 3]);  cat_1 = None
        expand_2: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 64, 128]);  permute_6 = None
        view_16: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_2, [768, 64, 128]);  expand_2 = None
        bmm: "f16[768, 64, 128]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm, [1, 12, 64, 64, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1258 in forward, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_11: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32)
        amax: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_11, [-1], True)
        abs_1: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax)
        eq: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq, full_default_1, amax);  eq = full_default_1 = amax = None
        sub_1: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_11, where);  convert_element_type_11 = None
        exp: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        log: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        add_2: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log, where);  log = where = None
        convert_element_type_12: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1259 in forward, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_2: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_17, convert_element_type_12);  view_17 = convert_element_type_12 = None
        exp_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_3: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_1, [1, 12, 64, 64, 128]);  exp_1 = None
        view_18: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_3, [768, 64, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1164 in forward, code: value_vectors = self.value(hidden_states)
        view_5: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 256]);  convert_element_type_1 = None
        permute_2: "f16[256, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "f16[4096, 768]" = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
        view_6: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 4096, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_9: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, 4096, 12, 64]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_5: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_12: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_5, [1, 12, 64, 64, 64]);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_3: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_12, 2, -1, 9223372036854775807)
        slice_4: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_12, 2, 0, -1)
        cat_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_3, slice_4], 2);  slice_3 = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_3: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_2, view_12], 3);  cat_2 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_4: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_3, [1, 12, 64, 128, 64]);  cat_3 = None
        view_19: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_4, [768, 128, 64]);  expand_4 = None
        bmm_1: "f16[768, 64, 64]" = torch.ops.aten.bmm.default(view_18, view_19);  view_18 = view_19 = None
        view_20: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 64, 64, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1275 in forward, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_21: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_20, [1, 12, 4096, 64]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_7: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_5: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_22: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 4096, 768]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_23: "f16[4096, 768]" = torch.ops.aten.reshape.default(view_22, [4096, 768]);  view_22 = None
        permute_8: "f16[768, 256]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_3: "f16[4096, 256]" = torch.ops.aten.mm.default(view_23, permute_8);  view_23 = permute_8 = None
        view_24: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(mm_3, [1, 4096, 256]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in forward, code: attn_output = prev_attn_output + attn_output
        add_3: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(getitem_1, view_24);  getitem_1 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_17: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 4096, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 4096, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_5);  convert_element_type_17 = getitem_5 = None
        add_4: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_1: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_2: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        add_5: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_3, arg8_1);  mul_3 = arg8_1 = None
        convert_element_type_18: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view_25: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_18, [4096, 256]);  convert_element_type_18 = None
        permute_9: "f16[256, 512]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm: "f16[4096, 512]" = torch.ops.aten.addmm.default(arg10_1, view_25, permute_9);  arg10_1 = view_25 = permute_9 = None
        view_26: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(addmm, [1, 4096, 512]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu: "f16[1, 4096, 512]" = torch.ops.aten.relu.default(view_26);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_27: "f16[4096, 512]" = torch.ops.aten.reshape.default(relu, [4096, 512]);  relu = None
        permute_10: "f16[512, 256]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f16[4096, 256]" = torch.ops.aten.addmm.default(arg12_1, view_27, permute_10);  arg12_1 = view_27 = permute_10 = None
        view_28: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_1, [1, 4096, 256]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_6: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(getitem, view_28);  getitem = view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_25: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_25, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 4096, 1]" = var_mean_2[0]
        getitem_7: "f32[1, 4096, 1]" = var_mean_2[1];  var_mean_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_4: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_25, getitem_7);  convert_element_type_25 = getitem_7 = None
        add_7: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_2: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_4: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_5: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_8: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        convert_element_type_26: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_8, torch.float16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:511 in forward, code: query_key_vectors = self.query_key(hidden_states)
        view_29: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_26, [4096, 256])
        permute_11: "f16[256, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_4: "f16[4096, 768]" = torch.ops.aten.mm.default(view_29, permute_11);  view_29 = permute_11 = None
        view_30: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_4, [1, 4096, 768]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_33: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_30, [1, 4096, 12, 64]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_13: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_1: "f16[1, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(permute_13, 4)
        unsqueeze_2: "f16[1, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 5);  unsqueeze_1 = None
        permute_15: "f16[1, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 4, 2, 5, 3]);  unsqueeze_2 = None
        permute_17: "f16[12, 4096, 64, 1, 1, 1]" = torch.ops.aten.permute.default(permute_15, [1, 3, 5, 0, 2, 4]);  permute_15 = None
        view_35: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_17, [12, 4096, 64]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:719 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_2: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
        convert_element_type_default_2: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_3: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 4);  convert_element_type_default_2 = None
        unsqueeze_4: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 5);  unsqueeze_3 = None
        permute_16: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_4, [4, 0, 2, 5, 3, 1]);  unsqueeze_4 = None
        permute_18: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_16, [1, 5, 0, 2, 4, 3]);  permute_16 = None
        view_36: "f16[12, 64, 64]" = torch.ops.aten.reshape.default(permute_18, [12, 64, 64]);  permute_18 = None
        bmm_2: "f16[12, 4096, 64]" = torch.ops.aten.bmm.default(view_35, view_36);  view_35 = view_36 = None
        view_37: "f16[12, 4096, 1, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm_2, [12, 4096, 1, 1, 1, 64]);  bmm_2 = None
        permute_19: "f16[1, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(view_37, [3, 0, 4, 1, 5, 2]);  view_37 = None
        view_38: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_19, [1, 12, 1, 4096, 64]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:724 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(view_38)
        cat_6: "f16[1, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([view_38, neg], -1);  view_38 = neg = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:725 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax: "i64[1, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_6, -1);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_6: "i64[1]" = torch.ops.aten.mul.Tensor(iota_1, 128);  iota_1 = None
        view_39: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_6, [1, 1, -1, 1]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:757 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_5: "i64[1, 12, 1, 1]" = torch.ops.aten.expand.default(view_39, [1, 12, 1, 1]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:758 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_9: "i64[1, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax, expand_5);  argmax = expand_5 = None
        view_40: "i64[1, 12, 4096]" = torch.ops.aten.reshape.default(add_9, [1, 12, 4096]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_7: "i64[1, 12, 4096]" = torch.ops.aten.mul.Tensor(view_40, 4096);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:153 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_2: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_41: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_2, [1, 1, -1]);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:154 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_6: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_41, [1, 12, 4096]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_6, 4096);  expand_6 = None
        add_10: "i64[1, 12, 4096]" = torch.ops.aten.add.Tensor(mul_7, remainder);  mul_7 = remainder = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:156 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort = torch.ops.aten.sort.default(add_10);  add_10 = None
        getitem_9: "i64[1, 12, 4096]" = sort[1];  sort = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        _tensor_constant1: "i64[]" = self._tensor_constant1
        lift_fresh_copy_1: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        convert_element_type_33: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_1, torch.float64);  lift_fresh_copy_1 = convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:563 in forward, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_1: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_9, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:400 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape)
        view_48: "i64[1, 12, 64, 64]" = torch.ops.aten.reshape.default(remainder_1, [1, 12, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_7: "i64[1, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_48, -1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_11: "i64[1, 12, 1, 64]" = torch.ops.aten.slice.Tensor(view_48, 2, -1, 9223372036854775807)
        slice_12: "i64[1, 12, 63, 64]" = torch.ops.aten.slice.Tensor(view_48, 2, 0, -1)
        cat_11: "i64[1, 12, 64, 64]" = torch.ops.aten.cat.default([slice_11, slice_12], 2);  slice_11 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_12: "i64[1, 12, 64, 128]" = torch.ops.aten.cat.default([cat_11, view_48], 3);  cat_11 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_8: "i64[1, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_12, -2);  cat_12 = None
        ne: "b8[1, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_7, unsqueeze_8);  unsqueeze_7 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_13, [1, 1, 1, 1]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_5: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_1, -1)
        expand_8: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_5, [-1, -1, -1, 64]);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_1, 2, expand_8);  repeat_1 = expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_43: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather, [1, 12, -1, 64, 64]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_10: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_43, [1, 12, 64, 64, 64])
        view_45: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(expand_10, [768, 64, 64]);  expand_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1054 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_1: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2)
        mean: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1055 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_11: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt_3: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_8: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_43, rsqrt_3);  view_43 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default_2: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1047 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div_1: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_8, full_default_2);  mul_8 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_7: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_1, 2, -1, 9223372036854775807)
        slice_8: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_1, 2, 0, -1)
        cat_7: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_7, slice_8], 2);  slice_7 = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_8: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_7, div_1], 3);  cat_7 = div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_20: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_8, [0, 1, 2, 4, 3]);  cat_8 = None
        expand_11: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_20, [1, 12, 64, 64, 128]);  permute_20 = None
        view_46: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_11, [768, 64, 128]);  expand_11 = None
        bmm_3: "f16[768, 64, 128]" = torch.ops.aten.bmm.default(view_45, view_46);  view_45 = view_46 = None
        view_47: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 64, 64, 128]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:880 in _attend, code: query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        where_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne, view_47, arg17_1);  ne = view_47 = arg17_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:885 in _attend, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_36: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32)
        amax_1: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_36, [-1], True)
        abs_2: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_1)
        eq_1: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_2, inf);  abs_2 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_1, full_default_3, amax_1);  eq_1 = full_default_3 = amax_1 = None
        sub_5: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_36, where_2);  convert_element_type_36 = None
        exp_2: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True);  exp_2 = None
        log_1: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_2);  sum_2 = None
        add_12: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_1, where_2);  log_1 = where_2 = None
        convert_element_type_37: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:887 in _attend, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_6: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_1, convert_element_type_37);  where_1 = convert_element_type_37 = None
        exp_3: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_12: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_3, [1, 12, 64, 64, 128]);  exp_3 = None
        view_49: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_12, [768, 64, 128]);  expand_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:512 in forward, code: value_vectors = self.value(hidden_states)
        view_31: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_26, [4096, 256]);  convert_element_type_26 = None
        permute_12: "f16[256, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_5: "f16[4096, 768]" = torch.ops.aten.mm.default(view_31, permute_12);  view_31 = permute_12 = None
        view_32: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_5, [1, 4096, 768]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_34: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_32, [1, 4096, 12, 64]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_14: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_2: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_14, [1, 1, 1, 1]);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_6: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_1, -1);  remainder_1 = None
        expand_9: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_6, [-1, -1, -1, 64]);  unsqueeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_2, 2, expand_9);  repeat_2 = expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_44: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_1, [1, 12, -1, 64, 64]);  gather_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_9: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_44, 2, -1, 9223372036854775807)
        slice_10: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_44, 2, 0, -1)
        cat_9: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_9, slice_10], 2);  slice_9 = slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_10: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_9, view_44], 3);  cat_9 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_13: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_10, [1, 12, 64, 128, 64]);  cat_10 = None
        view_50: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_13, [768, 128, 64]);  expand_13 = None
        bmm_4: "f16[768, 64, 64]" = torch.ops.aten.bmm.default(view_49, view_50);  view_49 = view_50 = None
        view_51: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 64, 64, 64]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:904 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_53: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_51, [1, 12, 4096, 64]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:776 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty: "i64[1, 12, 4096]" = torch.ops.aten.empty.memory_format([1, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:770 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_3: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:771 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        view_42: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_3, [1, 1, -1]);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:772 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_7: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_42, [1, 12, 4096]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:777 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter: "i64[1, 12, 4096]" = torch.ops.aten.scatter.src(empty, -1, getitem_9, expand_7);  empty = getitem_9 = expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1080 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_11: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter, -1);  scatter = None
        expand_15: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_11, [1, 12, 4096, 64]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_2: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(view_53, 2, expand_15);  view_53 = expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_21: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_2, [0, 2, 1, 3]);  gather_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_10: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_54: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 4096, 768]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_56: "f16[4096, 768]" = torch.ops.aten.reshape.default(view_54, [4096, 768]);  view_54 = None
        permute_22: "f16[768, 256]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_6: "f16[4096, 256]" = torch.ops.aten.mm.default(view_56, permute_22);  view_56 = permute_22 = None
        view_57: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(mm_6, [1, 4096, 256]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in forward, code: attn_output = prev_attn_output + attn_output
        add_13: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_3, view_57);  add_3 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_42: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_42, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 4096, 1]" = var_mean_3[0]
        getitem_11: "f32[1, 4096, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_7: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_11);  convert_element_type_42 = getitem_11 = None
        add_14: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_4: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_9: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_10: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        add_15: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        convert_element_type_43: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_15, torch.float16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view_58: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_43, [4096, 256]);  convert_element_type_43 = None
        permute_23: "f16[256, 512]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_2: "f16[4096, 512]" = torch.ops.aten.addmm.default(arg23_1, view_58, permute_23);  arg23_1 = view_58 = permute_23 = None
        view_59: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(addmm_2, [1, 4096, 512]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu_1: "f16[1, 4096, 512]" = torch.ops.aten.relu.default(view_59);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f16[4096, 512]" = torch.ops.aten.reshape.default(relu_1, [4096, 512]);  relu_1 = None
        permute_24: "f16[512, 256]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_3: "f16[4096, 256]" = torch.ops.aten.addmm.default(arg25_1, view_60, permute_24);  arg25_1 = view_60 = permute_24 = None
        view_61: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_3, [1, 4096, 256]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_16: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_6, view_61);  add_6 = view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_50: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 4096, 1]" = var_mean_4[0]
        getitem_13: "f32[1, 4096, 1]" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        _tensor_constant2: "i64[]" = self._tensor_constant2
        lift_fresh_copy_2: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        convert_element_type_58: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_2, torch.float64);  lift_fresh_copy_2 = convert_element_type_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_8: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_13);  convert_element_type_50 = getitem_13 = None
        add_17: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_5: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_11: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_12: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_11, arg26_1);  mul_11 = arg26_1 = None
        add_18: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_12, arg27_1);  mul_12 = arg27_1 = None
        convert_element_type_51: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_18, torch.float16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1162 in forward, code: query_vectors = self.query(hidden_states)
        view_62: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_51, [4096, 256])
        permute_25: "f16[256, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_7: "f16[4096, 768]" = torch.ops.aten.mm.default(view_62, permute_25);  view_62 = permute_25 = None
        view_63: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_7, [1, 4096, 768]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_68: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_63, [1, 4096, 12, 64]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_28: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_71: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_28, [1, 12, 64, 64, 64]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_16: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_71, [1, 12, 64, 64, 64]);  view_71 = None
        clone_14: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_76: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(clone_14, [768, 64, 64]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        view_64: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_51, [4096, 256])
        permute_26: "f16[256, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_8: "f16[4096, 768]" = torch.ops.aten.mm.default(view_64, permute_26);  view_64 = permute_26 = None
        view_65: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 4096, 768]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_69: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_65, [1, 4096, 12, 64]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_29: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        full_default_4: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_2: "f16[1, 12, 4096, 64]" = torch.ops.aten.div.Tensor(permute_29, full_default_4);  permute_29 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_72: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(div_2, [1, 12, 64, 64, 64]);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_13: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_72, 2, -1, 9223372036854775807)
        slice_14: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_72, 2, 0, -1)
        cat_13: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_13, slice_14], 2);  slice_13 = slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_14: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_13, view_72], 3);  cat_13 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_31: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_14, [0, 1, 2, 4, 3]);  cat_14 = None
        expand_17: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_31, [1, 12, 64, 64, 128]);  permute_31 = None
        view_77: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_17, [768, 64, 128]);  expand_17 = None
        bmm_5: "f16[768, 64, 128]" = torch.ops.aten.bmm.default(view_76, view_77);  view_76 = view_77 = None
        view_78: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 64, 64, 128]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1258 in forward, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_61: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32)
        amax_2: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_61, [-1], True)
        abs_3: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_2)
        eq_2: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_3, inf);  abs_3 = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_2, full_default_5, amax_2);  eq_2 = full_default_5 = amax_2 = None
        sub_9: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_61, where_3);  convert_element_type_61 = None
        exp_4: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True);  exp_4 = None
        log_2: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_3);  sum_3 = None
        add_19: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_2, where_3);  log_2 = where_3 = None
        convert_element_type_62: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1259 in forward, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_10: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_78, convert_element_type_62);  view_78 = convert_element_type_62 = None
        exp_5: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_18: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_5, [1, 12, 64, 64, 128]);  exp_5 = None
        view_79: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_18, [768, 64, 128]);  expand_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1164 in forward, code: value_vectors = self.value(hidden_states)
        view_66: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_51, [4096, 256]);  convert_element_type_51 = None
        permute_27: "f16[256, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_9: "f16[4096, 768]" = torch.ops.aten.mm.default(view_66, permute_27);  view_66 = permute_27 = None
        view_67: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_9, [1, 4096, 768]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_70: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_67, [1, 4096, 12, 64]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_30: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_73: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_30, [1, 12, 64, 64, 64]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_15: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_73, 2, -1, 9223372036854775807)
        slice_16: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_73, 2, 0, -1)
        cat_15: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_15, slice_16], 2);  slice_15 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_16: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_15, view_73], 3);  cat_15 = view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_19: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_16, [1, 12, 64, 128, 64]);  cat_16 = None
        view_80: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_19, [768, 128, 64]);  expand_19 = None
        bmm_6: "f16[768, 64, 64]" = torch.ops.aten.bmm.default(view_79, view_80);  view_79 = view_80 = None
        view_81: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 64, 64, 64]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1275 in forward, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_82: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_81, [1, 12, 4096, 64]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_32: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_16: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_83: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_16, [1, 4096, 768]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f16[4096, 768]" = torch.ops.aten.reshape.default(view_83, [4096, 768]);  view_83 = None
        permute_33: "f16[768, 256]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_10: "f16[4096, 256]" = torch.ops.aten.mm.default(view_84, permute_33);  view_84 = permute_33 = None
        view_85: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(mm_10, [1, 4096, 256]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in forward, code: attn_output = prev_attn_output + attn_output
        add_20: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_13, view_85);  add_13 = view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_67: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_67, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 4096, 1]" = var_mean_5[0]
        getitem_15: "f32[1, 4096, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_11: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_67, getitem_15);  convert_element_type_67 = getitem_15 = None
        add_21: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_6: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_13: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = rsqrt_6 = None
        mul_14: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_13, arg32_1);  mul_13 = arg32_1 = None
        add_22: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        convert_element_type_68: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_22, torch.float16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_68, [4096, 256]);  convert_element_type_68 = None
        permute_34: "f16[256, 512]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_4: "f16[4096, 512]" = torch.ops.aten.addmm.default(arg35_1, view_86, permute_34);  arg35_1 = view_86 = permute_34 = None
        view_87: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(addmm_4, [1, 4096, 512]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu_2: "f16[1, 4096, 512]" = torch.ops.aten.relu.default(view_87);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_88: "f16[4096, 512]" = torch.ops.aten.reshape.default(relu_2, [4096, 512]);  relu_2 = None
        permute_35: "f16[512, 256]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_5: "f16[4096, 256]" = torch.ops.aten.addmm.default(arg37_1, view_88, permute_35);  arg37_1 = view_88 = permute_35 = None
        view_89: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_5, [1, 4096, 256]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_23: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_16, view_89);  add_16 = view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_75: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_75, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 4096, 1]" = var_mean_6[0]
        getitem_17: "f32[1, 4096, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_12: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_75, getitem_17);  convert_element_type_75 = getitem_17 = None
        add_24: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_7: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_15: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
        mul_16: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_15, arg38_1);  mul_15 = arg38_1 = None
        add_25: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_16, arg39_1);  mul_16 = arg39_1 = None
        convert_element_type_76: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:511 in forward, code: query_key_vectors = self.query_key(hidden_states)
        view_90: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_76, [4096, 256])
        permute_36: "f16[256, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_11: "f16[4096, 768]" = torch.ops.aten.mm.default(view_90, permute_36);  view_90 = permute_36 = None
        view_91: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_11, [1, 4096, 768]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_94: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_91, [1, 4096, 12, 64]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_38: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_12: "f16[1, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(permute_38, 4)
        unsqueeze_13: "f16[1, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 5);  unsqueeze_12 = None
        permute_40: "f16[1, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_13, [0, 1, 4, 2, 5, 3]);  unsqueeze_13 = None
        permute_42: "f16[12, 4096, 64, 1, 1, 1]" = torch.ops.aten.permute.default(permute_40, [1, 3, 5, 0, 2, 4]);  permute_40 = None
        view_96: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_42, [12, 4096, 64]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:719 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_1: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default_1, 'randn');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_1: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_14: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 4);  convert_element_type_default_1 = None
        unsqueeze_15: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 5);  unsqueeze_14 = None
        permute_41: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_15, [4, 0, 2, 5, 3, 1]);  unsqueeze_15 = None
        permute_43: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_41, [1, 5, 0, 2, 4, 3]);  permute_41 = None
        view_97: "f16[12, 64, 64]" = torch.ops.aten.reshape.default(permute_43, [12, 64, 64]);  permute_43 = None
        bmm_7: "f16[12, 4096, 64]" = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
        view_98: "f16[12, 4096, 1, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm_7, [12, 4096, 1, 1, 1, 64]);  bmm_7 = None
        permute_44: "f16[1, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(view_98, [3, 0, 4, 1, 5, 2]);  view_98 = None
        view_99: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_44, [1, 12, 1, 4096, 64]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:724 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg_1: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(view_99)
        cat_19: "f16[1, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([view_99, neg_1], -1);  view_99 = neg_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:725 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax_1: "i64[1, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_19, -1);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_5: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_17: "i64[1]" = torch.ops.aten.mul.Tensor(iota_5, 128);  iota_5 = None
        view_100: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_17, [1, 1, -1, 1]);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:757 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_20: "i64[1, 12, 1, 1]" = torch.ops.aten.expand.default(view_100, [1, 12, 1, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:758 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_26: "i64[1, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_1, expand_20);  argmax_1 = expand_20 = None
        view_101: "i64[1, 12, 4096]" = torch.ops.aten.reshape.default(add_26, [1, 12, 4096]);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_18: "i64[1, 12, 4096]" = torch.ops.aten.mul.Tensor(view_101, 4096);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:153 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_6: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_102: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_6, [1, 1, -1]);  iota_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:154 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_21: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_102, [1, 12, 4096]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder_2: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_21, 4096);  expand_21 = None
        add_27: "i64[1, 12, 4096]" = torch.ops.aten.add.Tensor(mul_18, remainder_2);  mul_18 = remainder_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:156 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort_1 = torch.ops.aten.sort.default(add_27);  add_27 = None
        getitem_19: "i64[1, 12, 4096]" = sort_1[1];  sort_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        _tensor_constant3: "i64[]" = self._tensor_constant3
        lift_fresh_copy_3: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        convert_element_type_83: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_3, torch.float64);  lift_fresh_copy_3 = convert_element_type_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:563 in forward, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_3: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_19, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:400 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape)
        view_109: "i64[1, 12, 64, 64]" = torch.ops.aten.reshape.default(remainder_3, [1, 12, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_18: "i64[1, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_23: "i64[1, 12, 1, 64]" = torch.ops.aten.slice.Tensor(view_109, 2, -1, 9223372036854775807)
        slice_24: "i64[1, 12, 63, 64]" = torch.ops.aten.slice.Tensor(view_109, 2, 0, -1)
        cat_24: "i64[1, 12, 64, 64]" = torch.ops.aten.cat.default([slice_23, slice_24], 2);  slice_23 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_25: "i64[1, 12, 64, 128]" = torch.ops.aten.cat.default([cat_24, view_109], 3);  cat_24 = view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_19: "i64[1, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_25, -2);  cat_25 = None
        ne_1: "b8[1, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_18, unsqueeze_19);  unsqueeze_18 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_4: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_38, [1, 1, 1, 1]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_16: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_3, -1)
        expand_23: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_16, [-1, -1, -1, 64]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_4: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_4, 2, expand_23);  repeat_4 = expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_104: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_4, [1, 12, -1, 64, 64]);  gather_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_25: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_104, [1, 12, 64, 64, 64])
        view_106: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(expand_25, [768, 64, 64]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1054 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(view_104, 2)
        mean_1: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1055 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_28: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_8: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_19: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_104, rsqrt_8);  view_104 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default_6: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1047 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div_3: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_19, full_default_6);  mul_19 = full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_19: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_3, 2, -1, 9223372036854775807)
        slice_20: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_3, 2, 0, -1)
        cat_20: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_19, slice_20], 2);  slice_19 = slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_21: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_20, div_3], 3);  cat_20 = div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_45: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_21, [0, 1, 2, 4, 3]);  cat_21 = None
        expand_26: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_45, [1, 12, 64, 64, 128]);  permute_45 = None
        view_107: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_26, [768, 64, 128]);  expand_26 = None
        bmm_8: "f16[768, 64, 128]" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 64, 64, 128]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:880 in _attend, code: query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        where_4: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne_1, view_108, arg42_1);  ne_1 = view_108 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:885 in _attend, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_86: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(where_4, torch.float32)
        amax_3: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_86, [-1], True)
        abs_4: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_3)
        eq_3: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_4, inf);  abs_4 = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_3, full_default_7, amax_3);  eq_3 = full_default_7 = amax_3 = None
        sub_13: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_86, where_5);  convert_element_type_86 = None
        exp_6: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_4: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True);  exp_6 = None
        log_3: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        add_29: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_3, where_5);  log_3 = where_5 = None
        convert_element_type_87: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_29, torch.float16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:887 in _attend, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_14: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_4, convert_element_type_87);  where_4 = convert_element_type_87 = None
        exp_7: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_27: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_7, [1, 12, 64, 64, 128]);  exp_7 = None
        view_110: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_27, [768, 64, 128]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:512 in forward, code: value_vectors = self.value(hidden_states)
        view_92: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_76, [4096, 256]);  convert_element_type_76 = None
        permute_37: "f16[256, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_12: "f16[4096, 768]" = torch.ops.aten.mm.default(view_92, permute_37);  view_92 = permute_37 = None
        view_93: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_12, [1, 4096, 768]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_95: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_93, [1, 4096, 12, 64]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_39: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_5: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_39, [1, 1, 1, 1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_17: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_3, -1);  remainder_3 = None
        expand_24: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_17, [-1, -1, -1, 64]);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_5: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_5, 2, expand_24);  repeat_5 = expand_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_105: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_5, [1, 12, -1, 64, 64]);  gather_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_21: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_105, 2, -1, 9223372036854775807)
        slice_22: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_105, 2, 0, -1)
        cat_22: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_21, slice_22], 2);  slice_21 = slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_23: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_22, view_105], 3);  cat_22 = view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_28: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_23, [1, 12, 64, 128, 64]);  cat_23 = None
        view_111: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_28, [768, 128, 64]);  expand_28 = None
        bmm_9: "f16[768, 64, 64]" = torch.ops.aten.bmm.default(view_110, view_111);  view_110 = view_111 = None
        view_112: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 64, 64, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:904 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_114: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_112, [1, 12, 4096, 64]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:776 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty_1: "i64[1, 12, 4096]" = torch.ops.aten.empty.memory_format([1, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:770 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_7: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:771 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        view_103: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_7, [1, 1, -1]);  iota_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:772 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_22: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_103, [1, 12, 4096]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:777 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter_1: "i64[1, 12, 4096]" = torch.ops.aten.scatter.src(empty_1, -1, getitem_19, expand_22);  empty_1 = getitem_19 = expand_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1080 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_22: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_1, -1);  scatter_1 = None
        expand_30: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_22, [1, 12, 4096, 64]);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_6: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(view_114, 2, expand_30);  view_114 = expand_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_46: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_6, [0, 2, 1, 3]);  gather_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_21: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_115: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_21, [1, 4096, 768]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_117: "f16[4096, 768]" = torch.ops.aten.reshape.default(view_115, [4096, 768]);  view_115 = None
        permute_47: "f16[768, 256]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_13: "f16[4096, 256]" = torch.ops.aten.mm.default(view_117, permute_47);  view_117 = permute_47 = None
        view_118: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(mm_13, [1, 4096, 256]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in forward, code: attn_output = prev_attn_output + attn_output
        add_30: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_20, view_118);  add_20 = view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_92: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_92, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 4096, 1]" = var_mean_7[0]
        getitem_21: "f32[1, 4096, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_15: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_92, getitem_21);  convert_element_type_92 = getitem_21 = None
        add_31: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_9: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_20: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_21: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_32: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        convert_element_type_93: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view_119: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_93, [4096, 256]);  convert_element_type_93 = None
        permute_48: "f16[256, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_6: "f16[4096, 512]" = torch.ops.aten.addmm.default(arg48_1, view_119, permute_48);  arg48_1 = view_119 = permute_48 = None
        view_120: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(addmm_6, [1, 4096, 512]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu_3: "f16[1, 4096, 512]" = torch.ops.aten.relu.default(view_120);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_121: "f16[4096, 512]" = torch.ops.aten.reshape.default(relu_3, [4096, 512]);  relu_3 = None
        permute_49: "f16[512, 256]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_7: "f16[4096, 256]" = torch.ops.aten.addmm.default(arg50_1, view_121, permute_49);  arg50_1 = view_121 = permute_49 = None
        view_122: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_7, [1, 4096, 256]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_33: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_23, view_122);  add_23 = view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_100: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_100, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 4096, 1]" = var_mean_8[0]
        getitem_23: "f32[1, 4096, 1]" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        _tensor_constant4: "i64[]" = self._tensor_constant4
        lift_fresh_copy_4: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        convert_element_type_108: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_4, torch.float64);  lift_fresh_copy_4 = convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_16: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_100, getitem_23);  convert_element_type_100 = getitem_23 = None
        add_34: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_10: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_22: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_23: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        add_35: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None
        convert_element_type_101: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_35, torch.float16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1162 in forward, code: query_vectors = self.query(hidden_states)
        view_123: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_101, [4096, 256])
        permute_50: "f16[256, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_14: "f16[4096, 768]" = torch.ops.aten.mm.default(view_123, permute_50);  view_123 = permute_50 = None
        view_124: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_14, [1, 4096, 768]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_129: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_124, [1, 4096, 12, 64]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_53: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_132: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_53, [1, 12, 64, 64, 64]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_31: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_132, [1, 12, 64, 64, 64]);  view_132 = None
        clone_25: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_137: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(clone_25, [768, 64, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        view_125: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_101, [4096, 256])
        permute_51: "f16[256, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_15: "f16[4096, 768]" = torch.ops.aten.mm.default(view_125, permute_51);  view_125 = permute_51 = None
        view_126: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_15, [1, 4096, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_130: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_126, [1, 4096, 12, 64]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_54: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        full_default_8: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_4: "f16[1, 12, 4096, 64]" = torch.ops.aten.div.Tensor(permute_54, full_default_8);  permute_54 = full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_133: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(div_4, [1, 12, 64, 64, 64]);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_25: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_133, 2, -1, 9223372036854775807)
        slice_26: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_133, 2, 0, -1)
        cat_26: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_25, slice_26], 2);  slice_25 = slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_27: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_26, view_133], 3);  cat_26 = view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_56: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_27, [0, 1, 2, 4, 3]);  cat_27 = None
        expand_32: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_56, [1, 12, 64, 64, 128]);  permute_56 = None
        view_138: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_32, [768, 64, 128]);  expand_32 = None
        bmm_10: "f16[768, 64, 128]" = torch.ops.aten.bmm.default(view_137, view_138);  view_137 = view_138 = None
        view_139: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 64, 64, 128]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1258 in forward, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_111: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32)
        amax_4: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_111, [-1], True)
        abs_5: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_4)
        eq_4: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_5, inf);  abs_5 = None
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_4, full_default_9, amax_4);  eq_4 = full_default_9 = amax_4 = None
        sub_17: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_111, where_6);  convert_element_type_111 = None
        exp_8: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_5: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True);  exp_8 = None
        log_4: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_5);  sum_5 = None
        add_36: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_4, where_6);  log_4 = where_6 = None
        convert_element_type_112: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_36, torch.float16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1259 in forward, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_18: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_139, convert_element_type_112);  view_139 = convert_element_type_112 = None
        exp_9: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_33: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_9, [1, 12, 64, 64, 128]);  exp_9 = None
        view_140: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_33, [768, 64, 128]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1164 in forward, code: value_vectors = self.value(hidden_states)
        view_127: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_101, [4096, 256]);  convert_element_type_101 = None
        permute_52: "f16[256, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_16: "f16[4096, 768]" = torch.ops.aten.mm.default(view_127, permute_52);  view_127 = permute_52 = None
        view_128: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_16, [1, 4096, 768]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_131: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_128, [1, 4096, 12, 64]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_55: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_134: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_55, [1, 12, 64, 64, 64]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_27: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_134, 2, -1, 9223372036854775807)
        slice_28: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_134, 2, 0, -1)
        cat_28: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_27, slice_28], 2);  slice_27 = slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_29: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_28, view_134], 3);  cat_28 = view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_34: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_29, [1, 12, 64, 128, 64]);  cat_29 = None
        view_141: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_34, [768, 128, 64]);  expand_34 = None
        bmm_11: "f16[768, 64, 64]" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
        view_142: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 64, 64, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1275 in forward, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_143: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_142, [1, 12, 4096, 64]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_57: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_27: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_144: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_27, [1, 4096, 768]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_145: "f16[4096, 768]" = torch.ops.aten.reshape.default(view_144, [4096, 768]);  view_144 = None
        permute_58: "f16[768, 256]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_17: "f16[4096, 256]" = torch.ops.aten.mm.default(view_145, permute_58);  view_145 = permute_58 = None
        view_146: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(mm_17, [1, 4096, 256]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in forward, code: attn_output = prev_attn_output + attn_output
        add_37: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_30, view_146);  add_30 = view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_117: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_117, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 4096, 1]" = var_mean_9[0]
        getitem_25: "f32[1, 4096, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_19: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_117, getitem_25);  convert_element_type_117 = getitem_25 = None
        add_38: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_11: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_24: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_11);  sub_19 = rsqrt_11 = None
        mul_25: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_24, arg57_1);  mul_24 = arg57_1 = None
        add_39: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_25, arg58_1);  mul_25 = arg58_1 = None
        convert_element_type_118: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_39, torch.float16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view_147: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_118, [4096, 256]);  convert_element_type_118 = None
        permute_59: "f16[256, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_8: "f16[4096, 512]" = torch.ops.aten.addmm.default(arg60_1, view_147, permute_59);  arg60_1 = view_147 = permute_59 = None
        view_148: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(addmm_8, [1, 4096, 512]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu_4: "f16[1, 4096, 512]" = torch.ops.aten.relu.default(view_148);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_149: "f16[4096, 512]" = torch.ops.aten.reshape.default(relu_4, [4096, 512]);  relu_4 = None
        permute_60: "f16[512, 256]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_9: "f16[4096, 256]" = torch.ops.aten.addmm.default(arg62_1, view_149, permute_60);  arg62_1 = view_149 = permute_60 = None
        view_150: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_9, [1, 4096, 256]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_40: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_33, view_150);  add_33 = view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_125: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_40, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_125, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 4096, 1]" = var_mean_10[0]
        getitem_27: "f32[1, 4096, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_20: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_125, getitem_27);  convert_element_type_125 = getitem_27 = None
        add_41: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_12: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_26: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = rsqrt_12 = None
        mul_27: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_26, arg63_1);  mul_26 = arg63_1 = None
        add_42: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_27, arg64_1);  mul_27 = arg64_1 = None
        convert_element_type_126: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_42, torch.float16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:511 in forward, code: query_key_vectors = self.query_key(hidden_states)
        view_151: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_126, [4096, 256])
        permute_61: "f16[256, 768]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_18: "f16[4096, 768]" = torch.ops.aten.mm.default(view_151, permute_61);  view_151 = permute_61 = None
        view_152: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_18, [1, 4096, 768]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_155: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_152, [1, 4096, 12, 64]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_63: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_23: "f16[1, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(permute_63, 4)
        unsqueeze_24: "f16[1, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 5);  unsqueeze_23 = None
        permute_65: "f16[1, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_24, [0, 1, 4, 2, 5, 3]);  unsqueeze_24 = None
        permute_67: "f16[12, 4096, 64, 1, 1, 1]" = torch.ops.aten.permute.default(permute_65, [1, 3, 5, 0, 2, 4]);  permute_65 = None
        view_157: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_67, [12, 4096, 64]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:719 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default_2, 'randn');  inductor_lookup_seed_default_2 = None
        convert_element_type_default: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_25: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, 4);  convert_element_type_default = None
        unsqueeze_26: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 5);  unsqueeze_25 = None
        permute_66: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_26, [4, 0, 2, 5, 3, 1]);  unsqueeze_26 = None
        permute_68: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_66, [1, 5, 0, 2, 4, 3]);  permute_66 = None
        view_158: "f16[12, 64, 64]" = torch.ops.aten.reshape.default(permute_68, [12, 64, 64]);  permute_68 = None
        bmm_12: "f16[12, 4096, 64]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
        view_159: "f16[12, 4096, 1, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm_12, [12, 4096, 1, 1, 1, 64]);  bmm_12 = None
        permute_69: "f16[1, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(view_159, [3, 0, 4, 1, 5, 2]);  view_159 = None
        view_160: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_69, [1, 12, 1, 4096, 64]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:724 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg_2: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(view_160)
        cat_32: "f16[1, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([view_160, neg_2], -1);  view_160 = neg_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:725 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax_2: "i64[1, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_32, -1);  cat_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_9: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_28: "i64[1]" = torch.ops.aten.mul.Tensor(iota_9, 128);  iota_9 = None
        view_161: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_28, [1, 1, -1, 1]);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:757 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_35: "i64[1, 12, 1, 1]" = torch.ops.aten.expand.default(view_161, [1, 12, 1, 1]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:758 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_43: "i64[1, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_2, expand_35);  argmax_2 = expand_35 = None
        view_162: "i64[1, 12, 4096]" = torch.ops.aten.reshape.default(add_43, [1, 12, 4096]);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_29: "i64[1, 12, 4096]" = torch.ops.aten.mul.Tensor(view_162, 4096);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:153 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_10: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_163: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_10, [1, 1, -1]);  iota_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:154 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_36: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_163, [1, 12, 4096]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder_4: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_36, 4096);  expand_36 = None
        add_44: "i64[1, 12, 4096]" = torch.ops.aten.add.Tensor(mul_29, remainder_4);  mul_29 = remainder_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:156 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort_2 = torch.ops.aten.sort.default(add_44);  add_44 = None
        getitem_29: "i64[1, 12, 4096]" = sort_2[1];  sort_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        _tensor_constant5: "i64[]" = self._tensor_constant5
        lift_fresh_copy_5: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        convert_element_type_133: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_5, torch.float64);  lift_fresh_copy_5 = convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:563 in forward, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_5: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_29, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:400 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape)
        view_170: "i64[1, 12, 64, 64]" = torch.ops.aten.reshape.default(remainder_5, [1, 12, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_29: "i64[1, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_170, -1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_35: "i64[1, 12, 1, 64]" = torch.ops.aten.slice.Tensor(view_170, 2, -1, 9223372036854775807)
        slice_36: "i64[1, 12, 63, 64]" = torch.ops.aten.slice.Tensor(view_170, 2, 0, -1)
        cat_37: "i64[1, 12, 64, 64]" = torch.ops.aten.cat.default([slice_35, slice_36], 2);  slice_35 = slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_38: "i64[1, 12, 64, 128]" = torch.ops.aten.cat.default([cat_37, view_170], 3);  cat_37 = view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_30: "i64[1, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_38, -2);  cat_38 = None
        ne_2: "b8[1, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_29, unsqueeze_30);  unsqueeze_29 = unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_7: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_63, [1, 1, 1, 1]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_27: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_5, -1)
        expand_38: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_27, [-1, -1, -1, 64]);  unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_8: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_7, 2, expand_38);  repeat_7 = expand_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_165: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_8, [1, 12, -1, 64, 64]);  gather_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_40: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_165, [1, 12, 64, 64, 64])
        view_167: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(expand_40, [768, 64, 64]);  expand_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1054 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_3: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(view_165, 2)
        mean_2: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1055 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_45: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_13: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_30: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_165, rsqrt_13);  view_165 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default_10: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1047 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div_5: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_30, full_default_10);  mul_30 = full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_31: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_5, 2, -1, 9223372036854775807)
        slice_32: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_5, 2, 0, -1)
        cat_33: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_31, slice_32], 2);  slice_31 = slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_34: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_33, div_5], 3);  cat_33 = div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_70: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_34, [0, 1, 2, 4, 3]);  cat_34 = None
        expand_41: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_70, [1, 12, 64, 64, 128]);  permute_70 = None
        view_168: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_41, [768, 64, 128]);  expand_41 = None
        bmm_13: "f16[768, 64, 128]" = torch.ops.aten.bmm.default(view_167, view_168);  view_167 = view_168 = None
        view_169: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 64, 64, 128]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:880 in _attend, code: query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        where_7: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne_2, view_169, arg67_1);  ne_2 = view_169 = arg67_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:885 in _attend, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_136: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(where_7, torch.float32)
        amax_5: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_136, [-1], True)
        abs_6: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_5)
        eq_5: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_6, inf);  abs_6 = None
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_5, full_default_11, amax_5);  eq_5 = full_default_11 = amax_5 = None
        sub_21: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_136, where_8);  convert_element_type_136 = None
        exp_10: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_6: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True);  exp_10 = None
        log_5: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_6);  sum_6 = None
        add_46: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_5, where_8);  log_5 = where_8 = None
        convert_element_type_137: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_46, torch.float16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:887 in _attend, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_22: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_7, convert_element_type_137);  where_7 = convert_element_type_137 = None
        exp_11: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_42: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_11, [1, 12, 64, 64, 128]);  exp_11 = None
        view_171: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_42, [768, 64, 128]);  expand_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:512 in forward, code: value_vectors = self.value(hidden_states)
        view_153: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_126, [4096, 256]);  convert_element_type_126 = None
        permute_62: "f16[256, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_19: "f16[4096, 768]" = torch.ops.aten.mm.default(view_153, permute_62);  view_153 = permute_62 = None
        view_154: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_19, [1, 4096, 768]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_156: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_154, [1, 4096, 12, 64]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_64: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_8: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_64, [1, 1, 1, 1]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_28: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_5, -1);  remainder_5 = None
        expand_39: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_28, [-1, -1, -1, 64]);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_9: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_8, 2, expand_39);  repeat_8 = expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_166: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_9, [1, 12, -1, 64, 64]);  gather_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_33: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_166, 2, -1, 9223372036854775807)
        slice_34: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_166, 2, 0, -1)
        cat_35: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_33, slice_34], 2);  slice_33 = slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_36: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_35, view_166], 3);  cat_35 = view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_43: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_36, [1, 12, 64, 128, 64]);  cat_36 = None
        view_172: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_43, [768, 128, 64]);  expand_43 = None
        bmm_14: "f16[768, 64, 64]" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
        view_173: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 64, 64, 64]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:904 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_175: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_173, [1, 12, 4096, 64]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:776 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty_2: "i64[1, 12, 4096]" = torch.ops.aten.empty.memory_format([1, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:770 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_11: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:771 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        view_164: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_11, [1, 1, -1]);  iota_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:772 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_37: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_164, [1, 12, 4096]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:777 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter_2: "i64[1, 12, 4096]" = torch.ops.aten.scatter.src(empty_2, -1, getitem_29, expand_37);  empty_2 = getitem_29 = expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1080 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_33: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_2, -1);  scatter_2 = None
        expand_45: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_33, [1, 12, 4096, 64]);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_10: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(view_175, 2, expand_45);  view_175 = expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_71: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_10, [0, 2, 1, 3]);  gather_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_32: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_176: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_32, [1, 4096, 768]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_178: "f16[4096, 768]" = torch.ops.aten.reshape.default(view_176, [4096, 768]);  view_176 = None
        permute_72: "f16[768, 256]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_20: "f16[4096, 256]" = torch.ops.aten.mm.default(view_178, permute_72);  view_178 = permute_72 = None
        view_179: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(mm_20, [1, 4096, 256]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in forward, code: attn_output = prev_attn_output + attn_output
        add_47: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_37, view_179);  add_37 = view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_142: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_142, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 4096, 1]" = var_mean_11[0]
        getitem_31: "f32[1, 4096, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_23: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_142, getitem_31);  convert_element_type_142 = getitem_31 = None
        add_48: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_14: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_31: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = rsqrt_14 = None
        mul_32: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_31, arg70_1);  mul_31 = arg70_1 = None
        add_49: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_32, arg71_1);  mul_32 = arg71_1 = None
        convert_element_type_143: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view_180: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_143, [4096, 256]);  convert_element_type_143 = None
        permute_73: "f16[256, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_10: "f16[4096, 512]" = torch.ops.aten.addmm.default(arg73_1, view_180, permute_73);  arg73_1 = view_180 = permute_73 = None
        view_181: "f16[1, 4096, 512]" = torch.ops.aten.reshape.default(addmm_10, [1, 4096, 512]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu_5: "f16[1, 4096, 512]" = torch.ops.aten.relu.default(view_181);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_182: "f16[4096, 512]" = torch.ops.aten.reshape.default(relu_5, [4096, 512]);  relu_5 = None
        permute_74: "f16[512, 256]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_11: "f16[4096, 256]" = torch.ops.aten.addmm.default(arg75_1, view_182, permute_74);  arg75_1 = view_182 = permute_74 = None
        view_183: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_11, [1, 4096, 256]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_50: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_40, view_183);  add_40 = view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1703 in forward, code: return torch.cat([attn_output, hidden_states], dim=-1)
        cat_39: "f16[1, 4096, 512]" = torch.ops.aten.cat.default([add_47, add_50], -1);  add_47 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1796 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_150: "f32[1, 4096, 512]" = torch.ops.prims.convert_element_type.default(cat_39, torch.float32);  cat_39 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 4096, 1]" = var_mean_12[0]
        getitem_33: "f32[1, 4096, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_24: "f32[1, 4096, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_33);  convert_element_type_150 = getitem_33 = None
        add_51: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_15: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_33: "f32[1, 4096, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
        mul_34: "f32[1, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_33, arg76_1);  mul_33 = arg76_1 = None
        add_52: "f32[1, 4096, 512]" = torch.ops.aten.add.Tensor(mul_34, arg77_1);  mul_34 = arg77_1 = None
        convert_element_type_151: "f16[1, 4096, 512]" = torch.ops.prims.convert_element_type.default(add_52, torch.float16);  add_52 = None
        return (convert_element_type_151,)
