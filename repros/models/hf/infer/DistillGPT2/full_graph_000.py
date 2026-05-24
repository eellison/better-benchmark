import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 512]", arg1_1: "f32[50257, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304]", arg6_1: "f32[768, 2304]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304]", arg18_1: "f32[768, 2304]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304]", arg30_1: "f32[768, 2304]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304]", arg42_1: "f32[768, 2304]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304]", arg54_1: "f32[768, 2304]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304]", arg66_1: "f32[768, 2304]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  getitem_1 = None
        add_4: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
        addmm: "f32[16384, 2304]" = torch.ops.aten.addmm.default(arg5_1, view_1, arg6_1);  arg5_1 = view_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm, [32, 512, 2304]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "f32[32, 512, 768]" = split[0]
        getitem_3: "f32[32, 512, 768]" = split[1]
        getitem_4: "f32[32, 512, 768]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [32, 512, -1, 64]);  getitem_2 = None
        permute_2: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [32, 512, -1, 64]);  getitem_3 = None
        permute: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [32, 512, -1, 64]);  getitem_4 = None
        permute_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[512]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_8: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(unsqueeze, [32, -1]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[32, 513]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_3: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513)
        slice_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512);  cat = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[32, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[32, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[32, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[32, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, [32, -1, 512, 512]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_2: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where, [32, 12, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_2, False, scale = 0.125);  permute_2 = permute = permute_1 = expand_2 = None
        getitem_5: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_3, [32, 512, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg7_1, view_7, arg8_1);  arg7_1 = view_7 = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_8, add_1);  view_8 = add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_9: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_10: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_10);  getitem_10 = None
        add_7: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_8, [-1, 768]);  add_8 = None
        addmm_2: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg11_1, view_9, arg12_1);  arg11_1 = view_9 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_5);  view_10 = mul_5 = None
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_10);  mul_4 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg13_1, view_11, arg14_1);  arg13_1 = view_11 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, view_12);  add_6 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_11: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_12: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_12);  getitem_12 = None
        add_12: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        addmm_4: "f32[16384, 2304]" = torch.ops.aten.addmm.default(arg17_1, view_13, arg18_1);  arg17_1 = view_13 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 2304]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "f32[32, 512, 768]" = split_1[0]
        getitem_14: "f32[32, 512, 768]" = split_1[1]
        getitem_15: "f32[32, 512, 768]" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [32, 512, -1, 64]);  getitem_13 = None
        permute_6: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [32, 512, -1, 64]);  getitem_14 = None
        permute_4: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [32, 512, -1, 64]);  getitem_15 = None
        permute_5: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_3: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_1, [32, 12, 512, 512]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_3, False, scale = 0.125);  permute_6 = permute_4 = permute_5 = expand_3 = None
        getitem_16: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_7, [32, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg19_1, view_19, arg20_1);  arg19_1 = view_19 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_20, add_11);  view_20 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_21);  getitem_21 = None
        add_15: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_16: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_16, [-1, 768]);  add_16 = None
        addmm_6: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg23_1, view_21, arg24_1);  arg23_1 = view_21 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_13);  view_22 = mul_13 = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_18: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_18);  mul_12 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg25_1, view_23, arg26_1);  arg25_1 = view_23 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_14, view_24);  add_14 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  getitem_23 = None
        add_20: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_21: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
        addmm_8: "f32[16384, 2304]" = torch.ops.aten.addmm.default(arg29_1, view_25, arg30_1);  arg29_1 = view_25 = arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 2304]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "f32[32, 512, 768]" = split_2[0]
        getitem_25: "f32[32, 512, 768]" = split_2[1]
        getitem_26: "f32[32, 512, 768]" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [32, 512, -1, 64]);  getitem_24 = None
        permute_10: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [32, 512, -1, 64]);  getitem_25 = None
        permute_8: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [32, 512, -1, 64]);  getitem_26 = None
        permute_9: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_4: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_2, [32, 12, 512, 512]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_4, False, scale = 0.125);  permute_10 = permute_8 = permute_9 = expand_4 = None
        getitem_27: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_11, [32, 512, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg31_1, view_31, arg32_1);  arg31_1 = view_31 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_32, add_19);  view_32 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_31: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_32: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_32);  getitem_32 = None
        add_23: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_24: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_24, [-1, 768]);  add_24 = None
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_33, arg36_1);  arg35_1 = view_33 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_21: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_21);  view_34 = mul_21 = None
        mul_22: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_26: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_26);  mul_20 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg37_1, view_35, arg38_1);  arg37_1 = view_35 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_22, view_36);  add_22 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_33: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_34: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_34);  getitem_34 = None
        add_28: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
        addmm_12: "f32[16384, 2304]" = torch.ops.aten.addmm.default(arg41_1, view_37, arg42_1);  arg41_1 = view_37 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 2304]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "f32[32, 512, 768]" = split_3[0]
        getitem_36: "f32[32, 512, 768]" = split_3[1]
        getitem_37: "f32[32, 512, 768]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [32, 512, -1, 64]);  getitem_35 = None
        permute_14: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [32, 512, -1, 64]);  getitem_36 = None
        permute_12: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [32, 512, -1, 64]);  getitem_37 = None
        permute_13: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_5: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_3, [32, 12, 512, 512]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_5, False, scale = 0.125);  permute_14 = permute_12 = permute_13 = expand_5 = None
        getitem_38: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_15, [32, 512, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg43_1, view_43, arg44_1);  arg43_1 = view_43 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_44, add_27);  view_44 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_43);  getitem_43 = None
        add_31: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_32: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_32, [-1, 768]);  add_32 = None
        addmm_14: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg47_1, view_45, arg48_1);  arg47_1 = view_45 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_29: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_29);  view_46 = mul_29 = None
        mul_30: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_34);  mul_28 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg49_1, view_47, arg50_1);  arg49_1 = view_47 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_30, view_48);  add_30 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_10: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_45);  getitem_45 = None
        add_36: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
        addmm_16: "f32[16384, 2304]" = torch.ops.aten.addmm.default(arg53_1, view_49, arg54_1);  arg53_1 = view_49 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 2304]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "f32[32, 512, 768]" = split_4[0]
        getitem_47: "f32[32, 512, 768]" = split_4[1]
        getitem_48: "f32[32, 512, 768]" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [32, 512, -1, 64]);  getitem_46 = None
        permute_18: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [32, 512, -1, 64]);  getitem_47 = None
        permute_16: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [32, 512, -1, 64]);  getitem_48 = None
        permute_17: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_6: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_4, [32, 12, 512, 512]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_6, False, scale = 0.125);  permute_18 = permute_16 = permute_17 = expand_6 = None
        getitem_49: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_19, [32, 512, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg55_1, view_55, arg56_1);  arg55_1 = view_55 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_56, add_35);  view_56 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_53: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_54: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_54);  getitem_54 = None
        add_39: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_34: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_35: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_40: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_40, [-1, 768]);  add_40 = None
        addmm_18: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_57, arg60_1);  arg59_1 = view_57 = arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_37: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_37);  view_58 = mul_37 = None
        mul_38: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_42: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_42);  mul_36 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg61_1, view_59, arg62_1);  arg61_1 = view_59 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_55: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_56: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_56);  getitem_56 = None
        add_44: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_40: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
        addmm_20: "f32[16384, 2304]" = torch.ops.aten.addmm.default(arg65_1, view_61, arg66_1);  arg65_1 = view_61 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 2304]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "f32[32, 512, 768]" = split_5[0]
        getitem_58: "f32[32, 512, 768]" = split_5[1]
        getitem_59: "f32[32, 512, 768]" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [32, 512, -1, 64]);  getitem_57 = None
        permute_22: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [32, 512, -1, 64]);  getitem_58 = None
        permute_20: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [32, 512, -1, 64]);  getitem_59 = None
        permute_21: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  expand_1 = full_default_12 = full_default_11 = None
        expand_7: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_5, [32, 12, 512, 512]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_7, False, scale = 0.125);  permute_22 = permute_20 = permute_21 = expand_7 = None
        getitem_60: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_23, [32, 512, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg67_1, view_67, arg68_1);  arg67_1 = view_67 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_68, add_43);  view_68 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_65: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_13: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_65);  getitem_65 = None
        add_47: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_42: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_43: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_48: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_69, arg72_1);  arg71_1 = view_69 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_45: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_45);  view_70 = mul_45 = None
        mul_46: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_50: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_50);  mul_44 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg73_1, view_71, arg74_1);  arg73_1 = view_71 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_46, view_72);  add_46 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_67: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_67);  add_51 = getitem_67 = None
        add_52: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_48: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_74: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_53, [16384, 768])
        permute_24: "f32[768, 50257]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f32[768, 50260]" = torch.ops.aten.constant_pad_nd.default(permute_24, [0, 3, 0, 0]);  permute_24 = None
        mm_default: "f32[16384, 50260]" = torch.ops.aten.mm.default(view_74, constant_pad_nd_default);  view_74 = constant_pad_nd_default = None
        slice_tensor: "f32[16384, 50257]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_75: "f32[32, 512, 50257]" = torch.ops.aten.reshape.default(slice_tensor, [32, 512, 50257]);  slice_tensor = None
        return (view_75, add_53)
