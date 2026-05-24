import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[256]", arg1_1: "f32[256]", arg2_1: "f32[8, 4096, 256]", arg3_1: "f32[768, 256]", arg4_1: "f32[768, 256]", arg5_1: "f32[]", arg6_1: "f32[256, 768]", arg7_1: "f32[8, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 4096, 1]" = var_mean[0]
        getitem_1: "f32[8, 4096, 1]" = var_mean[1];  var_mean = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub: "f32[8, 4096, 256]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  arg2_1 = getitem_1 = None
        add: "f32[8, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 4096, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_1: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:511 in forward, code: query_key_vectors = self.query_key(hidden_states)
        view: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, [32768, 256])
        permute: "f32[256, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f32[32768, 768]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm, [8, 4096, 768]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_4: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_1, [8, 4096, 12, 64]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_2: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze: "f32[8, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(permute_2, 4)
        unsqueeze_1: "f32[8, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 5);  unsqueeze = None
        permute_4: "f32[8, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_1, [0, 1, 4, 2, 5, 3]);  unsqueeze_1 = None
        permute_6: "f32[12, 8, 4096, 64, 1, 1]" = torch.ops.aten.permute.default(permute_4, [1, 0, 3, 5, 2, 4]);  permute_4 = None
        view_6: "f32[12, 32768, 64]" = torch.ops.aten.reshape.default(permute_6, [12, 32768, 64]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:719 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_2: "f32[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(inductor_random_default_1, 4);  inductor_random_default_1 = None
        unsqueeze_3: "f32[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 5);  unsqueeze_2 = None
        view_7: "f32[12, 64, 64]" = torch.ops.aten.reshape.default(unsqueeze_3, [12, 64, 64]);  unsqueeze_3 = None
        bmm: "f32[12, 32768, 64]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8: "f32[12, 8, 4096, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm, [12, 8, 4096, 1, 1, 64]);  bmm = None
        permute_8: "f32[8, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(view_8, [1, 0, 4, 2, 5, 3]);  view_8 = None
        view_9: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_8, [8, 12, 1, 4096, 64]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:724 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(view_9)
        cat: "f32[8, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([view_9, neg], -1);  view_9 = neg = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:725 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax: "i64[8, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat, -1);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_2: "i64[1]" = torch.ops.aten.mul.Tensor(iota, 128);  iota = None
        view_10: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_2, [1, 1, -1, 1]);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:757 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand: "i64[8, 12, 1, 1]" = torch.ops.aten.expand.default(view_10, [8, 12, 1, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:758 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_2: "i64[8, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax, expand);  argmax = expand = None
        view_11: "i64[8, 12, 4096]" = torch.ops.aten.reshape.default(add_2, [8, 12, 4096]);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_3: "i64[8, 12, 4096]" = torch.ops.aten.mul.Tensor(view_11, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:153 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_1: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_12: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_1, [1, 1, -1]);  iota_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:154 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_1: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(view_12, [8, 12, 4096]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_1, 4096);  expand_1 = None
        add_3: "i64[8, 12, 4096]" = torch.ops.aten.add.Tensor(mul_3, remainder);  mul_3 = remainder = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:156 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort = torch.ops.aten.sort.default(add_3);  add_3 = None
        getitem_3: "i64[8, 12, 4096]" = sort[1];  sort = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        _tensor_constant0: "i64[]" = self._tensor_constant0
        lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        convert_element_type: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy, torch.float64);  lift_fresh_copy = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:668 in forward, code: buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes, -1)
        view_26: "i64[8, 12, 1, 4096]" = torch.ops.aten.reshape.default(view_11, [8, 12, 1, -1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1332 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:563 in forward, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_1: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_3, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:400 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape)
        view_19: "i64[8, 12, 64, 64]" = torch.ops.aten.reshape.default(remainder_1, [8, 12, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_6: "i64[8, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_5: "i64[8, 12, 1, 64]" = torch.ops.aten.slice.Tensor(view_19, 2, -1, 9223372036854775807)
        slice_6: "i64[8, 12, 63, 64]" = torch.ops.aten.slice.Tensor(view_19, 2, 0, -1)
        cat_5: "i64[8, 12, 64, 64]" = torch.ops.aten.cat.default([slice_5, slice_6], 2);  slice_5 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_6: "i64[8, 12, 64, 128]" = torch.ops.aten.cat.default([cat_5, view_19], 3);  cat_5 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_7: "i64[8, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_6, -2);  cat_6 = None
        ne: "b8[8, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_6, unsqueeze_7);  unsqueeze_6 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat: "f32[8, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_2, [1, 1, 1, 1]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_4: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_1, -1)
        expand_3: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_4, [-1, -1, -1, 64]);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat, 2, expand_3);  repeat = expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_14: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather, [8, 12, -1, 64, 64]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_5: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_14, [8, 12, 64, 64, 64])
        view_16: "f32[6144, 64, 64]" = torch.ops.aten.reshape.default(expand_5, [6144, 64, 64]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1054 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_1: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(view_14, 2)
        mean: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1055 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_4: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_14, rsqrt_1);  view_14 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1047 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_4, full_default);  mul_4 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_1: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div, 2, -1, 9223372036854775807)
        slice_2: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div, 2, 0, -1)
        cat_1: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_1, slice_2], 2);  slice_1 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_2: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_1, div], 3);  cat_1 = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_9: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_2, [0, 1, 2, 4, 3]);  cat_2 = None
        expand_6: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_9, [8, 12, 64, 64, 128]);  permute_9 = None
        view_17: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_6, [6144, 64, 128]);  expand_6 = None
        bmm_1: "f32[6144, 64, 128]" = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
        view_18: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 64, 64, 128]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:880 in _attend, code: query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        where: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne, view_18, arg5_1);  ne = view_18 = arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:885 in _attend, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        amax: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.amax.default(where, [-1], True)
        abs_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax)
        eq: "b8[8, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq, full_default_1, amax);  eq = full_default_1 = amax = None
        sub_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where, where_1)
        exp: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        log: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        add_5: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log, where_1);  log = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:887 in _attend, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where, add_5);  where = add_5 = None
        exp_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_7: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_1, [8, 12, 64, 64, 128]);  exp_1 = None
        view_20: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_7, [6144, 64, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:512 in forward, code: value_vectors = self.value(hidden_states)
        view_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, [32768, 256]);  add_1 = None
        permute_1: "f32[256, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f32[32768, 768]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm_1, [8, 4096, 768]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        view_5: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(view_3, [8, 4096, 12, 64]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_3: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_3, [1, 1, 1, 1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_5: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_1, -1);  remainder_1 = None
        expand_4: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_5, [-1, -1, -1, 64]);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_1, 2, expand_4);  repeat_1 = expand_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        view_15: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_1, [8, 12, -1, 64, 64]);  gather_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_3: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_15, 2, -1, 9223372036854775807)
        slice_4: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_15, 2, 0, -1)
        cat_3: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_3, slice_4], 2);  slice_3 = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_4: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_3, view_15], 3);  cat_3 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_8: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_4, [8, 12, 64, 128, 64]);  cat_4 = None
        view_21: "f32[6144, 128, 64]" = torch.ops.aten.reshape.default(expand_8, [6144, 128, 64]);  expand_8 = None
        bmm_2: "f32[6144, 64, 64]" = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = view_21 = None
        view_22: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_2, [8, 12, 64, 64, 64]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:904 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        view_24: "f32[8, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_22, [8, 12, 4096, 64]);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:776 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty: "i64[8, 12, 4096]" = torch.ops.aten.empty.memory_format([8, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:770 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_2: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:771 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        view_13: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_2, [1, 1, -1]);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:772 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_2: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(view_13, [8, 12, 4096]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:777 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter: "i64[8, 12, 4096]" = torch.ops.aten.scatter.src(empty, -1, getitem_3, expand_2);  empty = getitem_3 = expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1080 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_10: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter, -1);  scatter = None
        expand_10: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_10, [8, 12, 4096, 64]);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_2: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(view_24, 2, expand_10);  view_24 = expand_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_10: "f32[8, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_2, [0, 2, 1, 3]);  gather_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_1: "f32[8, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_25: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(clone_1, [8, 4096, 768]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        view_27: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_25, [32768, 768]);  view_25 = None
        permute_11: "f32[768, 256]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "f32[32768, 256]" = torch.ops.aten.mm.default(view_27, permute_11);  view_27 = permute_11 = None
        view_28: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(mm_2, [8, 4096, 256]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1332 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_5: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt, view_28);  gt = view_28 = None
        mul_6: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_5, 1.0526315789473684);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1556 in torch_dynamo_resume_in_forward_at_1541, code: attn_output = prev_attn_output + attn_output
        add_6: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(arg7_1, mul_6);  arg7_1 = None
        return (mul_6, view_26, add_6)
