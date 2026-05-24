import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "f32[50257, 768]", primals_3: "f32[1024, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[2304]", primals_7: "f32[768, 2304]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[3072]", primals_13: "f32[768, 3072]", primals_14: "f32[768]", primals_15: "f32[3072, 768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[2304]", primals_19: "f32[768, 2304]", primals_20: "f32[768]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[3072]", primals_25: "f32[768, 3072]", primals_26: "f32[768]", primals_27: "f32[3072, 768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[2304]", primals_31: "f32[768, 2304]", primals_32: "f32[768]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[3072]", primals_37: "f32[768, 3072]", primals_38: "f32[768]", primals_39: "f32[3072, 768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[2304]", primals_43: "f32[768, 2304]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[3072]", primals_49: "f32[768, 3072]", primals_50: "f32[768]", primals_51: "f32[3072, 768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[2304]", primals_55: "f32[768, 2304]", primals_56: "f32[768]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[3072]", primals_61: "f32[768, 3072]", primals_62: "f32[768]", primals_63: "f32[3072, 768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[2304]", primals_67: "f32[768, 2304]", primals_68: "f32[768]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[3072]", primals_73: "f32[768, 3072]", primals_74: "f32[768]", primals_75: "f32[3072, 768]", primals_76: "f32[768]", primals_77: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(unsqueeze, [32, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[32, 513]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512)
        slice_3: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513);  cat = None
        sub_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[32, 512]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[32, 512]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[32, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[32, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[32, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[32, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, [32, -1, 512, 512]);  bitwise_and_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_12: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 1e-30);  inductor_random_default_12 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 1.0);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(mul_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_4);  mul_2 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
        addmm: "f32[16384, 2304]" = torch.ops.aten.addmm.default(primals_6, view_1, primals_7);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm, [32, 512, 2304]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "f32[32, 512, 768]" = split[0]
        getitem_3: "f32[32, 512, 768]" = split[1]
        getitem_4: "f32[32, 512, 768]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [32, 512, -1, 64]);  getitem_3 = None
        permute: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [32, 512, -1, 64]);  getitem_4 = None
        permute_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [32, 512, -1, 64]);  getitem_2 = None
        permute_2: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  expand_1 = full_default_2 = full_default_1 = None
        expand_2: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where, [32, 12, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_2, True, 1e-30, scale = 0.125)
        getitem_5: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_6: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention[1]
        getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_3, [32, 512, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_8, view_7, primals_9);  primals_8 = view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_11: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 1e-30);  inductor_random_default_11 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_8);  view_8 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, 1.0);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, mul_1);  mul_5 = mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_9: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_10: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_10);  getitem_10 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, primals_10)
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_11);  mul_7 = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_8, [-1, 768]);  add_8 = None
        addmm_2: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_12, view_9, primals_13);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_9);  view_10 = mul_9 = None
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_11: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_8, add_10);  mul_8 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_11, [-1, 3072]);  mul_11 = None
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_14, view_11, primals_15);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_10: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 1e-30);  inductor_random_default_10 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_12);  view_12 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, 1.0);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, mul_13);  add_6 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_11: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_12: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_12);  getitem_12 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_16)
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_17);  mul_15 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        addmm_4: "f32[16384, 2304]" = torch.ops.aten.addmm.default(primals_18, view_13, primals_19);  primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 2304]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "f32[32, 512, 768]" = split_1[0]
        getitem_14: "f32[32, 512, 768]" = split_1[1]
        getitem_15: "f32[32, 512, 768]" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [32, 512, -1, 64]);  getitem_14 = None
        permute_4: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [32, 512, -1, 64]);  getitem_15 = None
        permute_5: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [32, 512, -1, 64]);  getitem_13 = None
        permute_6: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, True, 1e-30, scale = 0.125)
        getitem_16: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_17: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_7, [32, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_20, view_19, primals_21);  primals_20 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_9: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 1e-30);  inductor_random_default_9 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_20);  view_20 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, 1.0);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, add_11);  mul_17 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_15: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_21);  getitem_21 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_22)
        add_16: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_23);  mul_19 = primals_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_16, [-1, 768]);  add_16 = None
        addmm_6: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_24, view_21, primals_25);  primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_21: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_21);  view_22 = mul_21 = None
        mul_22: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_23: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_18);  mul_20 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_26, view_23, primals_27);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_8: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 1e-30);  inductor_random_default_8 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_4, view_24);  view_24 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, 1.0);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_14, mul_25);  add_14 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  getitem_23 = None
        mul_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_28)
        add_21: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_29);  mul_27 = primals_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
        addmm_8: "f32[16384, 2304]" = torch.ops.aten.addmm.default(primals_30, view_25, primals_31);  primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 2304]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "f32[32, 512, 768]" = split_2[0]
        getitem_25: "f32[32, 512, 768]" = split_2[1]
        getitem_26: "f32[32, 512, 768]" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [32, 512, -1, 64]);  getitem_25 = None
        permute_8: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [32, 512, -1, 64]);  getitem_26 = None
        permute_9: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [32, 512, -1, 64]);  getitem_24 = None
        permute_10: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, True, 1e-30, scale = 0.125)
        getitem_27: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_28: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_11, [32, 512, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_32, view_31, primals_33);  primals_32 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_7: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 1e-30);  inductor_random_default_7 = None
        mul_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_32);  view_32 = None
        mul_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, 1.0);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, add_19);  mul_29 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_31: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_32: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_32);  getitem_32 = None
        mul_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_34)
        add_24: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_35);  mul_31 = primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_24, [-1, 768]);  add_24 = None
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_36, view_33, primals_37);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_33: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_33);  view_34 = mul_33 = None
        mul_34: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_35: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_35, [-1, 3072]);  mul_35 = None
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_38, view_35, primals_39);  primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_6: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 1e-30);  inductor_random_default_6 = None
        mul_36: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_36);  view_36 = None
        mul_37: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, 1.0);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_22, mul_37);  add_22 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_33: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_34: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_34);  getitem_34 = None
        mul_38: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_39: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_40)
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_41);  mul_39 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
        addmm_12: "f32[16384, 2304]" = torch.ops.aten.addmm.default(primals_42, view_37, primals_43);  primals_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 2304]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "f32[32, 512, 768]" = split_3[0]
        getitem_36: "f32[32, 512, 768]" = split_3[1]
        getitem_37: "f32[32, 512, 768]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [32, 512, -1, 64]);  getitem_36 = None
        permute_12: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [32, 512, -1, 64]);  getitem_37 = None
        permute_13: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [32, 512, -1, 64]);  getitem_35 = None
        permute_14: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_2, True, 1e-30, scale = 0.125)
        getitem_38: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_39: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_15, [32, 512, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_44, view_43, primals_45);  primals_44 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_5: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 1e-30);  inductor_random_default_5 = None
        mul_40: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_7, view_44);  view_44 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, 1.0);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, add_27);  mul_41 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_31: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_43);  getitem_43 = None
        mul_42: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_43: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_46)
        add_32: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_47);  mul_43 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_32, [-1, 768]);  add_32 = None
        addmm_14: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_48, view_45, primals_49);  primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_45: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_45);  view_46 = mul_45 = None
        mul_46: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_47: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_34);  mul_44 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_50, view_47, primals_51);  primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_4: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 1e-30);  inductor_random_default_4 = None
        mul_48: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_48);  view_48 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, 1.0);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_30, mul_49);  add_30 = mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_45);  getitem_45 = None
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_51: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_52)
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_53);  mul_51 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
        addmm_16: "f32[16384, 2304]" = torch.ops.aten.addmm.default(primals_54, view_49, primals_55);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 2304]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "f32[32, 512, 768]" = split_4[0]
        getitem_47: "f32[32, 512, 768]" = split_4[1]
        getitem_48: "f32[32, 512, 768]" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [32, 512, -1, 64]);  getitem_47 = None
        permute_16: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [32, 512, -1, 64]);  getitem_48 = None
        permute_17: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [32, 512, -1, 64]);  getitem_46 = None
        permute_18: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_2, True, 1e-30, scale = 0.125)
        getitem_49: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_50: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_19, [32, 512, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_56, view_55, primals_57);  primals_56 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_3: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 1e-30);  inductor_random_default_3 = None
        mul_52: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_56);  view_56 = None
        mul_53: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, 1.0);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, add_35);  mul_53 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_53: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_54: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_39: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_54);  getitem_54 = None
        mul_54: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
        mul_55: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_58)
        add_40: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_59);  mul_55 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_40, [-1, 768]);  add_40 = None
        addmm_18: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_60, view_57, primals_61);  primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_57);  view_58 = mul_57 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_59: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_56, add_42);  mul_56 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_59, [-1, 3072]);  mul_59 = None
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_62, view_59, primals_63);  primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_2: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 1e-30);  inductor_random_default_2 = None
        mul_60: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_10, view_60);  view_60 = None
        mul_61: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, 1.0);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_38, mul_61);  add_38 = mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_55: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_56: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_56);  getitem_56 = None
        mul_62: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, primals_64)
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_63, primals_65);  mul_63 = primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
        addmm_20: "f32[16384, 2304]" = torch.ops.aten.addmm.default(primals_66, view_61, primals_67);  primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 2304]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "f32[32, 512, 768]" = split_5[0]
        getitem_58: "f32[32, 512, 768]" = split_5[1]
        getitem_59: "f32[32, 512, 768]" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [32, 512, -1, 64]);  getitem_58 = None
        permute_20: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [32, 512, -1, 64]);  getitem_59 = None
        permute_21: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [32, 512, -1, 64]);  getitem_57 = None
        permute_22: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_2, True, 1e-30, scale = 0.125);  expand_2 = None
        getitem_60: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_61: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_23, [32, 512, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_68, view_67, primals_69);  primals_68 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_1: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 1e-30);  inductor_random_default_1 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_68);  view_68 = None
        mul_65: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, 1.0);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, add_43);  mul_65 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_65: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_47: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_13: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_65);  getitem_65 = None
        mul_66: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
        mul_67: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_70)
        add_48: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_71);  mul_67 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_72, view_69, primals_73);  primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_69: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_69);  view_70 = mul_69 = None
        mul_70: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_71: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_50);  mul_68 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_74, view_71, primals_75);  primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_72: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_72);  view_72 = None
        mul_73: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, 1.0);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_46, mul_73);  add_46 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_67: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_67);  add_51 = getitem_67 = None
        mul_74: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
        mul_75: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_76)
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_77);  mul_75 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_24: "f32[768, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view_74: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_53, [16384, 768])
        constant_pad_nd_default_3: "f32[768, 50260]" = torch.ops.aten.constant_pad_nd.default(permute_24, [0, 3, 0, 0]);  permute_24 = None
        mm_default_2: "f32[16384, 50260]" = torch.ops.aten.mm.default(view_74, constant_pad_nd_default_3);  constant_pad_nd_default_3 = None
        slice_tensor_1: "f32[16384, 50257]" = torch.ops.aten.slice.Tensor(mm_default_2, 1, 0, -3);  mm_default_2 = None
        view_75: "f32[32, 512, 50257]" = torch.ops.aten.reshape.default(slice_tensor_1, [32, 512, 50257]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_30: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_71, [1, 0]);  view_71 = None
        permute_32: "f32[768, 16384]" = torch.ops.aten.permute.default(view_69, [1, 0]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_40: "f32[768, 16384]" = torch.ops.aten.permute.default(view_61, [1, 0]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_42: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_44: "f32[768, 16384]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_3: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_52: "f32[768, 16384]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_4: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_54: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_47, [1, 0]);  view_47 = None
        permute_56: "f32[768, 16384]" = torch.ops.aten.permute.default(view_45, [1, 0]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_5: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_64: "f32[768, 16384]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_6: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_66: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_68: "f32[768, 16384]" = torch.ops.aten.permute.default(view_33, [1, 0]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_7: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_76: "f32[768, 16384]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_8: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_78: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_23, [1, 0]);  view_23 = None
        permute_80: "f32[768, 16384]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_9: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_88: "f32[768, 16384]" = torch.ops.aten.permute.default(view_13, [1, 0]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_10: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_90: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        permute_92: "f32[768, 16384]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_11: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_100: "f32[768, 16384]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        return (view_75, add_53, primals_1, primals_2, primals_4, primals_7, primals_9, primals_10, primals_13, primals_15, primals_16, primals_19, primals_21, primals_22, primals_25, primals_27, primals_28, primals_31, primals_33, primals_34, primals_37, primals_39, primals_40, primals_43, primals_45, primals_46, primals_49, primals_51, primals_52, primals_55, primals_57, primals_58, primals_61, primals_63, primals_64, primals_67, primals_69, primals_70, primals_73, primals_75, primals_76, embedding, unsqueeze, embedding_1, gt, getitem_1, rsqrt, permute, permute_1, permute_2, where, getitem_5, getitem_6, getitem_7, getitem_8, gt_1, mul_6, addmm_2, gt_2, mul_14, permute_4, permute_5, permute_6, getitem_16, getitem_17, getitem_18, getitem_19, gt_3, mul_18, addmm_6, gt_4, mul_26, permute_8, permute_9, permute_10, getitem_27, getitem_28, getitem_29, getitem_30, gt_5, mul_30, addmm_10, gt_6, mul_38, permute_12, permute_13, permute_14, getitem_38, getitem_39, getitem_40, getitem_41, gt_7, mul_42, addmm_14, gt_8, mul_50, permute_16, permute_17, permute_18, getitem_49, getitem_50, getitem_51, getitem_52, gt_9, mul_54, addmm_18, gt_10, mul_62, permute_20, permute_21, permute_22, getitem_60, getitem_61, getitem_62, getitem_63, gt_11, mul_66, addmm_22, gt_12, mul_74, view_74, div, permute_30, permute_32, div_1, permute_40, div_2, permute_42, permute_44, div_3, permute_52, div_4, permute_54, permute_56, div_5, permute_64, div_6, permute_66, permute_68, div_7, permute_76, div_8, permute_78, permute_80, div_9, permute_88, div_10, permute_90, permute_92, div_11, permute_100)
