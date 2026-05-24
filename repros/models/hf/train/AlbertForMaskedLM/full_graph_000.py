import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 512]", primals_2: "i64[1, 512]", primals_3: "i64[1, 512]", primals_4: "f32[30000, 128]", primals_5: "f32[2, 128]", primals_6: "f32[512, 128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[4096, 128]", primals_10: "f32[4096]", primals_11: "f32[4096, 4096]", primals_12: "f32[4096]", primals_13: "f32[4096, 4096]", primals_14: "f32[4096]", primals_15: "f32[4096, 4096]", primals_16: "f32[4096]", primals_17: "f32[4096, 4096]", primals_18: "f32[4096]", primals_19: "f32[4096]", primals_20: "f32[4096]", primals_21: "f32[16384, 4096]", primals_22: "f32[16384]", primals_23: "f32[4096, 16384]", primals_24: "f32[4096]", primals_25: "f32[4096]", primals_26: "f32[4096]", primals_27: "f32[128, 4096]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[30000]", primals_32: "i64[8, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:94 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(primals_3, [1, -1]);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:95 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, primals_2);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:96 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[8, 512]" = torch.ops.aten.expand.default(gather, [8, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:101 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[8, 512, 128]" = torch.ops.aten.embedding.default(primals_4, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:102 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[8, 512, 128]" = torch.ops.aten.embedding.default(primals_5, expand_1);  primals_5 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:103 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:105 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_6, primals_2);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:106 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, primals_7)
        add_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[8, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [8, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        view: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_3, [4096, 128]);  add_3 = None
        permute: "f32[128, 4096]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_10, view, permute);  primals_10 = permute = None
        view_1: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm, [8, 512, 4096]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_2: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_1, [4096, 4096])
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_1: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_2, permute_1)
        view_3: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_1, [8, 512, 4096]);  addmm_1 = None
        view_4: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_3, [8, 512, -1, 64]);  view_3 = None
        permute_2: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_3: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_2: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_2, permute_3)
        view_6: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_2, [8, 512, 4096]);  addmm_2 = None
        view_7: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_6, [8, 512, -1, 64]);  view_6 = None
        permute_4: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_5: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_3: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_2, permute_5)
        view_9: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_3, [8, 512, 4096]);  addmm_3 = None
        view_10: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_9, [8, 512, -1, 64]);  view_9 = None
        permute_6: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  expand_2 = full_default = None
        mul_2: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_2, 0.3535533905932738);  permute_2 = None
        permute_7: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_3: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_7, 0.3535533905932738);  permute_7 = None
        expand_3: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_2, [8, 64, 512, 64]);  mul_2 = None
        clone_1: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_11: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_1, [512, 512, 64]);  clone_1 = None
        expand_4: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_3, [8, 64, 64, 512]);  mul_3 = None
        clone_2: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_12: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_2, [512, 64, 512]);  clone_2 = None
        bmm: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12)
        view_13: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm, [8, 64, 512, 512]);  bmm = None
        add_6: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = None
        amax: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  amax = None
        exp: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        eq: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_6, -inf);  add_6 = None
        logical_not: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[8, 64, 512, 512]" = torch.ops.aten.full.default([8, 64, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = div = None
        expand_5: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_1, [8, 64, 512, 512])
        view_14: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [512, 512, 512]);  expand_5 = None
        expand_6: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_6, [8, 64, 512, 64]);  permute_6 = None
        clone_3: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_15: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [512, 512, 64]);  clone_3 = None
        bmm_1: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = None
        view_16: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 64, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_4: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_17: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_4, [8, 512, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_18: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_17, [4096, 4096]);  view_17 = None
        permute_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_4: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_18, permute_9)
        view_19: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_4, [8, 512, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_7: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_1, view_19);  view_1 = view_19 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_8: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_7, getitem_3);  add_7 = getitem_3 = None
        mul_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_4, primals_19)
        add_9: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_5, primals_20);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_20: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_9, [4096, 4096])
        permute_10: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_5: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_20, permute_10)
        view_21: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_5, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        pow_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
        mul_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_21, mul_7);  view_21 = mul_7 = None
        mul_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_11: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_11);  mul_6 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_22: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_9, [4096, 16384]);  mul_9 = None
        permute_11: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_6: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_22, permute_11)
        view_23: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_6, [8, 512, 4096]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_12: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_23, add_9);  view_23 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_13: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_12, getitem_5);  add_12 = getitem_5 = None
        mul_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_10, primals_25)
        add_14: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_11, primals_26);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_24: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_14, [4096, 4096])
        addmm_7: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_24, permute_1)
        view_25: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_7, [8, 512, 4096]);  addmm_7 = None
        view_26: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_25, [8, 512, -1, 64]);  view_25 = None
        permute_13: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_8: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_24, permute_3)
        view_28: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_8, [8, 512, 4096]);  addmm_8 = None
        view_29: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_28, [8, 512, -1, 64]);  view_28 = None
        permute_15: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_9: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_24, permute_5)
        view_31: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_9, [8, 512, 4096]);  addmm_9 = None
        view_32: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_31, [8, 512, -1, 64]);  view_31 = None
        permute_17: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_12: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_13, 0.3535533905932738);  permute_13 = None
        permute_18: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_13: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_18, 0.3535533905932738);  permute_18 = None
        expand_7: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_12, [8, 64, 512, 64]);  mul_12 = None
        clone_6: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_33: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_6, [512, 512, 64]);  clone_6 = None
        expand_8: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_13, [8, 64, 64, 512]);  mul_13 = None
        clone_7: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_34: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_7, [512, 64, 512]);  clone_7 = None
        bmm_2: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34)
        view_35: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [8, 64, 512, 512]);  bmm_2 = None
        add_15: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_35, where);  view_35 = None
        amax_1: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_15, [-1], True)
        sub_4: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_15, amax_1);  amax_1 = None
        exp_1: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        eq_1: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_15, -inf);  add_15 = None
        logical_not_2: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_2, div_1);  logical_not_3 = div_1 = None
        expand_9: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_3, [8, 64, 512, 512])
        view_36: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [512, 512, 512]);  expand_9 = None
        expand_10: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_17, [8, 64, 512, 64]);  permute_17 = None
        clone_8: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_37: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [512, 512, 64]);  clone_8 = None
        bmm_3: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = None
        view_38: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 64, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_9: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_39: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_9, [8, 512, -1]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_40: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_39, [4096, 4096]);  view_39 = None
        addmm_10: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_40, permute_9)
        view_41: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_10, [8, 512, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_16: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_14, view_41);  add_14 = view_41 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_17: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_5: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_16, getitem_7);  add_16 = getitem_7 = None
        mul_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_14, primals_19)
        add_18: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_15, primals_20);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_42: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_18, [4096, 4096])
        addmm_11: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_42, permute_10)
        view_43: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_11, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        pow_2: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
        mul_17: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_43, mul_17);  view_43 = mul_17 = None
        mul_18: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_20: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_20);  mul_16 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_44: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_19, [4096, 16384]);  mul_19 = None
        addmm_12: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_44, permute_11)
        view_45: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_12, [8, 512, 4096]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_21: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_45, add_18);  view_45 = add_18 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_22: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_6: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_21, getitem_9);  add_21 = getitem_9 = None
        mul_20: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_21: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_20, primals_25)
        add_23: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_21, primals_26);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_46: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_23, [4096, 4096])
        addmm_13: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_46, permute_1)
        view_47: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_13, [8, 512, 4096]);  addmm_13 = None
        view_48: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_47, [8, 512, -1, 64]);  view_47 = None
        permute_24: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_14: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_46, permute_3)
        view_50: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_14, [8, 512, 4096]);  addmm_14 = None
        view_51: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_50, [8, 512, -1, 64]);  view_50 = None
        permute_26: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_15: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_46, permute_5)
        view_53: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_15, [8, 512, 4096]);  addmm_15 = None
        view_54: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_53, [8, 512, -1, 64]);  view_53 = None
        permute_28: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_22: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_24, 0.3535533905932738);  permute_24 = None
        permute_29: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_23: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_29, 0.3535533905932738);  permute_29 = None
        expand_11: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_22, [8, 64, 512, 64]);  mul_22 = None
        clone_11: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_55: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_11, [512, 512, 64]);  clone_11 = None
        expand_12: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_23, [8, 64, 64, 512]);  mul_23 = None
        clone_12: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_56: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_12, [512, 64, 512]);  clone_12 = None
        bmm_4: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56)
        view_57: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [8, 64, 512, 512]);  bmm_4 = None
        add_24: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_57, where);  view_57 = None
        amax_2: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_24, [-1], True)
        sub_7: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_24, amax_2);  amax_2 = None
        exp_2: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        eq_2: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_24, -inf);  add_24 = None
        logical_not_4: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_2, div_2);  logical_not_5 = div_2 = None
        expand_13: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_5, [8, 64, 512, 512])
        view_58: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [512, 512, 512]);  expand_13 = None
        expand_14: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_28, [8, 64, 512, 64]);  permute_28 = None
        clone_13: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_59: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_13, [512, 512, 64]);  clone_13 = None
        bmm_5: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = None
        view_60: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 64, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_14: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_14, [8, 512, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_62: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_61, [4096, 4096]);  view_61 = None
        addmm_16: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_62, permute_9)
        view_63: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_16, [8, 512, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_25: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_23, view_63);  add_23 = view_63 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_25, getitem_11);  add_25 = getitem_11 = None
        mul_24: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_25: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_24, primals_19)
        add_27: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_25, primals_20);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_64: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_27, [4096, 4096])
        addmm_17: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_64, permute_10)
        view_65: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_17, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        pow_3: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_27: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_65, mul_27);  view_65 = mul_27 = None
        mul_28: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_29: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_29);  mul_26 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_66: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_29, [4096, 16384]);  mul_29 = None
        addmm_18: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_66, permute_11)
        view_67: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_18, [8, 512, 4096]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_67, add_27);  view_67 = add_27 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_9: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_30, getitem_13);  add_30 = getitem_13 = None
        mul_30: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_31: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_30, primals_25)
        add_32: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_31, primals_26);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_68: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_32, [4096, 4096])
        addmm_19: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_68, permute_1)
        view_69: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_19, [8, 512, 4096]);  addmm_19 = None
        view_70: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_69, [8, 512, -1, 64]);  view_69 = None
        permute_35: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_20: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_68, permute_3)
        view_72: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_20, [8, 512, 4096]);  addmm_20 = None
        view_73: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_72, [8, 512, -1, 64]);  view_72 = None
        permute_37: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_21: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_68, permute_5)
        view_75: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_21, [8, 512, 4096]);  addmm_21 = None
        view_76: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_75, [8, 512, -1, 64]);  view_75 = None
        permute_39: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_32: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_35, 0.3535533905932738);  permute_35 = None
        permute_40: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        mul_33: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_40, 0.3535533905932738);  permute_40 = None
        expand_15: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_32, [8, 64, 512, 64]);  mul_32 = None
        clone_16: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_77: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [512, 512, 64]);  clone_16 = None
        expand_16: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_33, [8, 64, 64, 512]);  mul_33 = None
        clone_17: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_78: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_17, [512, 64, 512]);  clone_17 = None
        bmm_6: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
        view_79: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [8, 64, 512, 512]);  bmm_6 = None
        add_33: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_79, where);  view_79 = None
        amax_3: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_10: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_33, amax_3);  amax_3 = None
        exp_3: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        eq_3: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_33, -inf);  add_33 = None
        logical_not_6: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_2, div_3);  logical_not_7 = div_3 = None
        expand_17: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_7, [8, 64, 512, 512])
        view_80: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [512, 512, 512]);  expand_17 = None
        expand_18: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_39, [8, 64, 512, 64]);  permute_39 = None
        clone_18: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_81: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [512, 512, 64]);  clone_18 = None
        bmm_7: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = None
        view_82: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 64, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_19: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_83: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_19, [8, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_84: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_83, [4096, 4096]);  view_83 = None
        addmm_22: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_84, permute_9)
        view_85: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_22, [8, 512, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_34: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_32, view_85);  add_32 = view_85 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_35: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_11: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_34, getitem_15);  add_34 = getitem_15 = None
        mul_34: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_35: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_34, primals_19)
        add_36: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_35, primals_20);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_86: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_36, [4096, 4096])
        addmm_23: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_86, permute_10)
        view_87: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_23, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        pow_4: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
        mul_37: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_87, mul_37);  view_87 = mul_37 = None
        mul_38: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_38: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_38);  mul_36 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_88: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_39, [4096, 16384]);  mul_39 = None
        addmm_24: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_88, permute_11)
        view_89: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_24, [8, 512, 4096]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_39: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_89, add_36);  view_89 = add_36 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_40: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_12: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_39, getitem_17);  add_39 = getitem_17 = None
        mul_40: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_41: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, primals_25)
        add_41: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_41, primals_26);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_90: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_41, [4096, 4096])
        addmm_25: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_90, permute_1)
        view_91: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_25, [8, 512, 4096]);  addmm_25 = None
        view_92: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_91, [8, 512, -1, 64]);  view_91 = None
        permute_46: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_26: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_90, permute_3)
        view_94: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_26, [8, 512, 4096]);  addmm_26 = None
        view_95: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_94, [8, 512, -1, 64]);  view_94 = None
        permute_48: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_27: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_90, permute_5)
        view_97: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_27, [8, 512, 4096]);  addmm_27 = None
        view_98: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_97, [8, 512, -1, 64]);  view_97 = None
        permute_50: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_42: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_46, 0.3535533905932738);  permute_46 = None
        permute_51: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        mul_43: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_51, 0.3535533905932738);  permute_51 = None
        expand_19: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_42, [8, 64, 512, 64]);  mul_42 = None
        clone_21: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_99: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_21, [512, 512, 64]);  clone_21 = None
        expand_20: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_43, [8, 64, 64, 512]);  mul_43 = None
        clone_22: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_100: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_22, [512, 64, 512]);  clone_22 = None
        bmm_8: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100)
        view_101: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [8, 64, 512, 512]);  bmm_8 = None
        add_42: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_101, where);  view_101 = None
        amax_4: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
        sub_13: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_42, amax_4);  amax_4 = None
        exp_4: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        eq_4: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_42, -inf);  add_42 = None
        logical_not_8: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_2, div_4);  logical_not_9 = div_4 = None
        expand_21: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_9, [8, 64, 512, 512])
        view_102: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [512, 512, 512]);  expand_21 = None
        expand_22: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_50, [8, 64, 512, 64]);  permute_50 = None
        clone_23: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_103: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_23, [512, 512, 64]);  clone_23 = None
        bmm_9: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = None
        view_104: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 64, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_24: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_24, [8, 512, -1]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_106: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_105, [4096, 4096]);  view_105 = None
        addmm_28: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_106, permute_9)
        view_107: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_28, [8, 512, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_43: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_41, view_107);  add_41 = view_107 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_44: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_14: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_43, getitem_19);  add_43 = getitem_19 = None
        mul_44: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_45: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_44, primals_19)
        add_45: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_45, primals_20);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_108: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_45, [4096, 4096])
        addmm_29: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_108, permute_10)
        view_109: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_29, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        pow_5: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
        mul_47: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_109, mul_47);  view_109 = mul_47 = None
        mul_48: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_47: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_47);  mul_46 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_110: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_49, [4096, 16384]);  mul_49 = None
        addmm_30: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_110, permute_11)
        view_111: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_30, [8, 512, 4096]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_48: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_111, add_45);  view_111 = add_45 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_49: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_15: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_48, getitem_21);  add_48 = getitem_21 = None
        mul_50: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_51: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_50, primals_25)
        add_50: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_51, primals_26);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_112: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_50, [4096, 4096])
        addmm_31: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_112, permute_1)
        view_113: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_31, [8, 512, 4096]);  addmm_31 = None
        view_114: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_113, [8, 512, -1, 64]);  view_113 = None
        permute_57: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_32: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_112, permute_3)
        view_116: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_32, [8, 512, 4096]);  addmm_32 = None
        view_117: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_116, [8, 512, -1, 64]);  view_116 = None
        permute_59: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_33: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_112, permute_5)
        view_119: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_33, [8, 512, 4096]);  addmm_33 = None
        view_120: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_119, [8, 512, -1, 64]);  view_119 = None
        permute_61: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_52: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_57, 0.3535533905932738);  permute_57 = None
        permute_62: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        mul_53: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_62, 0.3535533905932738);  permute_62 = None
        expand_23: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_52, [8, 64, 512, 64]);  mul_52 = None
        clone_26: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_121: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [512, 512, 64]);  clone_26 = None
        expand_24: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_53, [8, 64, 64, 512]);  mul_53 = None
        clone_27: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_122: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_27, [512, 64, 512]);  clone_27 = None
        bmm_10: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122)
        view_123: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [8, 64, 512, 512]);  bmm_10 = None
        add_51: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_123, where);  view_123 = None
        amax_5: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_51, [-1], True)
        sub_16: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_51, amax_5);  amax_5 = None
        exp_5: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        eq_5: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_51, -inf);  add_51 = None
        logical_not_10: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_2, div_5);  logical_not_11 = div_5 = None
        expand_25: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_11, [8, 64, 512, 512])
        view_124: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [512, 512, 512]);  expand_25 = None
        expand_26: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_61, [8, 64, 512, 64]);  permute_61 = None
        clone_28: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_125: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_28, [512, 512, 64]);  clone_28 = None
        bmm_11: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = None
        view_126: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 64, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_29: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_127: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_29, [8, 512, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_128: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_127, [4096, 4096]);  view_127 = None
        addmm_34: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_128, permute_9)
        view_129: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_34, [8, 512, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_52: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_50, view_129);  add_50 = view_129 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_53: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_17: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_52, getitem_23);  add_52 = getitem_23 = None
        mul_54: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_55: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_54, primals_19)
        add_54: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_55, primals_20);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_130: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_54, [4096, 4096])
        addmm_35: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_130, permute_10)
        view_131: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_35, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        pow_6: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
        mul_57: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_131, mul_57);  view_131 = mul_57 = None
        mul_58: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_56: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_132: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_59, [4096, 16384]);  mul_59 = None
        addmm_36: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_132, permute_11)
        view_133: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_36, [8, 512, 4096]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_57: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_133, add_54);  view_133 = add_54 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_58: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_18: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_57, getitem_25);  add_57 = getitem_25 = None
        mul_60: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_61: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_60, primals_25)
        add_59: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_61, primals_26);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_134: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_59, [4096, 4096])
        addmm_37: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_134, permute_1)
        view_135: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_37, [8, 512, 4096]);  addmm_37 = None
        view_136: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_135, [8, 512, -1, 64]);  view_135 = None
        permute_68: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_38: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_134, permute_3)
        view_138: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_38, [8, 512, 4096]);  addmm_38 = None
        view_139: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_138, [8, 512, -1, 64]);  view_138 = None
        permute_70: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_39: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_134, permute_5)
        view_141: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_39, [8, 512, 4096]);  addmm_39 = None
        view_142: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_141, [8, 512, -1, 64]);  view_141 = None
        permute_72: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_62: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_68, 0.3535533905932738);  permute_68 = None
        permute_73: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        mul_63: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_73, 0.3535533905932738);  permute_73 = None
        expand_27: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_62, [8, 64, 512, 64]);  mul_62 = None
        clone_31: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_143: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_31, [512, 512, 64]);  clone_31 = None
        expand_28: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_63, [8, 64, 64, 512]);  mul_63 = None
        clone_32: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_144: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_32, [512, 64, 512]);  clone_32 = None
        bmm_12: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144)
        view_145: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [8, 64, 512, 512]);  bmm_12 = None
        add_60: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_145, where);  view_145 = None
        amax_6: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
        sub_19: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_60, amax_6);  amax_6 = None
        exp_6: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        eq_6: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_60, -inf);  add_60 = None
        logical_not_12: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_13: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_2, div_6);  logical_not_13 = div_6 = None
        expand_29: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_13, [8, 64, 512, 512])
        view_146: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [512, 512, 512]);  expand_29 = None
        expand_30: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_72, [8, 64, 512, 64]);  permute_72 = None
        clone_33: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_147: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_33, [512, 512, 64]);  clone_33 = None
        bmm_13: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = None
        view_148: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [8, 64, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_34: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_149: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_34, [8, 512, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_150: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_149, [4096, 4096]);  view_149 = None
        addmm_40: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_150, permute_9)
        view_151: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_40, [8, 512, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_61: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_59, view_151);  add_59 = view_151 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_62: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_20: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_61, getitem_27);  add_61 = getitem_27 = None
        mul_64: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_65: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_64, primals_19)
        add_63: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_65, primals_20);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_152: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_63, [4096, 4096])
        addmm_41: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_152, permute_10)
        view_153: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_41, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
        pow_7: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
        mul_67: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_153, mul_67);  view_153 = mul_67 = None
        mul_68: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_65: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_65);  mul_66 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_154: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_69, [4096, 16384]);  mul_69 = None
        addmm_42: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_154, permute_11)
        view_155: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_42, [8, 512, 4096]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_66: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_155, add_63);  view_155 = add_63 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_67: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_21: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_66, getitem_29);  add_66 = getitem_29 = None
        mul_70: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_71: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_70, primals_25)
        add_68: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_71, primals_26);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_156: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_68, [4096, 4096])
        addmm_43: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_156, permute_1)
        view_157: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_43, [8, 512, 4096]);  addmm_43 = None
        view_158: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_157, [8, 512, -1, 64]);  view_157 = None
        permute_79: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_44: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_156, permute_3)
        view_160: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_44, [8, 512, 4096]);  addmm_44 = None
        view_161: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_160, [8, 512, -1, 64]);  view_160 = None
        permute_81: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_45: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_156, permute_5)
        view_163: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_45, [8, 512, 4096]);  addmm_45 = None
        view_164: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_163, [8, 512, -1, 64]);  view_163 = None
        permute_83: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_72: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_79, 0.3535533905932738);  permute_79 = None
        permute_84: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        mul_73: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
        expand_31: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_72, [8, 64, 512, 64]);  mul_72 = None
        clone_36: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_165: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_36, [512, 512, 64]);  clone_36 = None
        expand_32: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_73, [8, 64, 64, 512]);  mul_73 = None
        clone_37: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_166: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_37, [512, 64, 512]);  clone_37 = None
        bmm_14: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166)
        view_167: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [8, 64, 512, 512]);  bmm_14 = None
        add_69: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_167, where);  view_167 = None
        amax_7: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_22: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_69, amax_7);  amax_7 = None
        exp_7: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        eq_7: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_69, -inf);  add_69 = None
        logical_not_14: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_15: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_2, div_7);  logical_not_15 = div_7 = None
        expand_33: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_15, [8, 64, 512, 512])
        view_168: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [512, 512, 512]);  expand_33 = None
        expand_34: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_83, [8, 64, 512, 64]);  permute_83 = None
        clone_38: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_169: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_38, [512, 512, 64]);  clone_38 = None
        bmm_15: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = None
        view_170: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 64, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_39: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_171: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_39, [8, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_172: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_171, [4096, 4096]);  view_171 = None
        addmm_46: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_172, permute_9)
        view_173: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_46, [8, 512, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_70: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_68, view_173);  add_68 = view_173 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_71: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_23: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_70, getitem_31);  add_70 = getitem_31 = None
        mul_74: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_75: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_74, primals_19)
        add_72: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_75, primals_20);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_174: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_72, [4096, 4096])
        addmm_47: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_174, permute_10)
        view_175: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_47, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        pow_8: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
        mul_77: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_175, mul_77);  view_175 = mul_77 = None
        mul_78: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_74: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_74);  mul_76 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_176: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_79, [4096, 16384]);  mul_79 = None
        addmm_48: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_176, permute_11)
        view_177: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_48, [8, 512, 4096]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_75: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_177, add_72);  view_177 = add_72 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_76: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_24: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_75, getitem_33);  add_75 = getitem_33 = None
        mul_80: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_81: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_80, primals_25)
        add_77: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_81, primals_26);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_178: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_77, [4096, 4096])
        addmm_49: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_178, permute_1)
        view_179: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_49, [8, 512, 4096]);  addmm_49 = None
        view_180: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_179, [8, 512, -1, 64]);  view_179 = None
        permute_90: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_50: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_178, permute_3)
        view_182: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_50, [8, 512, 4096]);  addmm_50 = None
        view_183: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_182, [8, 512, -1, 64]);  view_182 = None
        permute_92: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_51: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_178, permute_5)
        view_185: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_51, [8, 512, 4096]);  addmm_51 = None
        view_186: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_185, [8, 512, -1, 64]);  view_185 = None
        permute_94: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_82: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_90, 0.3535533905932738);  permute_90 = None
        permute_95: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        mul_83: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_95, 0.3535533905932738);  permute_95 = None
        expand_35: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_82, [8, 64, 512, 64]);  mul_82 = None
        clone_41: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_187: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_41, [512, 512, 64]);  clone_41 = None
        expand_36: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_83, [8, 64, 64, 512]);  mul_83 = None
        clone_42: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_188: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_42, [512, 64, 512]);  clone_42 = None
        bmm_16: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188)
        view_189: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [8, 64, 512, 512]);  bmm_16 = None
        add_78: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_189, where);  view_189 = None
        amax_8: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_25: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_8);  amax_8 = None
        exp_8: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        eq_8: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf);  add_78 = None
        logical_not_16: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_17: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_2, div_8);  logical_not_17 = div_8 = None
        expand_37: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_17, [8, 64, 512, 512])
        view_190: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [512, 512, 512]);  expand_37 = None
        expand_38: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_94, [8, 64, 512, 64]);  permute_94 = None
        clone_43: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_191: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_43, [512, 512, 64]);  clone_43 = None
        bmm_17: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = None
        view_192: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [8, 64, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_44: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_44, [8, 512, -1]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_194: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_193, [4096, 4096]);  view_193 = None
        addmm_52: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_194, permute_9)
        view_195: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_52, [8, 512, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_79: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_77, view_195);  add_77 = view_195 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_80: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_26: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_79, getitem_35);  add_79 = getitem_35 = None
        mul_84: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_85: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_84, primals_19)
        add_81: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_85, primals_20);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_196: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_81, [4096, 4096])
        addmm_53: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_196, permute_10)
        view_197: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_53, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        pow_9: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
        mul_87: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_197, mul_87);  view_197 = mul_87 = None
        mul_88: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_83: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_83);  mul_86 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_198: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_89, [4096, 16384]);  mul_89 = None
        addmm_54: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_198, permute_11)
        view_199: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_54, [8, 512, 4096]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_84: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_199, add_81);  view_199 = add_81 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_85: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_27: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_84, getitem_37);  add_84 = getitem_37 = None
        mul_90: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_91: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_90, primals_25)
        add_86: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_91, primals_26);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_200: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_86, [4096, 4096])
        addmm_55: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_200, permute_1)
        view_201: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_55, [8, 512, 4096]);  addmm_55 = None
        view_202: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_201, [8, 512, -1, 64]);  view_201 = None
        permute_101: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_56: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_200, permute_3)
        view_204: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_56, [8, 512, 4096]);  addmm_56 = None
        view_205: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_204, [8, 512, -1, 64]);  view_204 = None
        permute_103: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_57: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_200, permute_5)
        view_207: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_57, [8, 512, 4096]);  addmm_57 = None
        view_208: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_207, [8, 512, -1, 64]);  view_207 = None
        permute_105: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_92: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_101, 0.3535533905932738);  permute_101 = None
        permute_106: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        mul_93: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_106, 0.3535533905932738);  permute_106 = None
        expand_39: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_92, [8, 64, 512, 64]);  mul_92 = None
        clone_46: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_209: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_46, [512, 512, 64]);  clone_46 = None
        expand_40: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_93, [8, 64, 64, 512]);  mul_93 = None
        clone_47: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_210: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_47, [512, 64, 512]);  clone_47 = None
        bmm_18: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210)
        view_211: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [8, 64, 512, 512]);  bmm_18 = None
        add_87: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_211, where);  view_211 = None
        amax_9: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_87, [-1], True)
        sub_28: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_87, amax_9);  amax_9 = None
        exp_9: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        eq_9: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_87, -inf);  add_87 = None
        logical_not_18: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_19: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_2, div_9);  logical_not_19 = div_9 = None
        expand_41: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_19, [8, 64, 512, 512])
        view_212: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [512, 512, 512]);  expand_41 = None
        expand_42: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_105, [8, 64, 512, 64]);  permute_105 = None
        clone_48: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_213: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_48, [512, 512, 64]);  clone_48 = None
        bmm_19: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = None
        view_214: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 64, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_49: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_49, [8, 512, -1]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_216: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_215, [4096, 4096]);  view_215 = None
        addmm_58: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_216, permute_9)
        view_217: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_58, [8, 512, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_88: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_86, view_217);  add_86 = view_217 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_89: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_29: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_88, getitem_39);  add_88 = getitem_39 = None
        mul_94: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_95: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_94, primals_19)
        add_90: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_95, primals_20);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_218: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_90, [4096, 4096])
        addmm_59: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_218, permute_10)
        view_219: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_59, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        pow_10: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
        mul_97: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_219, mul_97);  view_219 = mul_97 = None
        mul_98: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_92: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_92);  mul_96 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_220: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_99, [4096, 16384]);  mul_99 = None
        addmm_60: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_220, permute_11)
        view_221: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_60, [8, 512, 4096]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_93: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_221, add_90);  view_221 = add_90 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_94: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_30: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_93, getitem_41);  add_93 = getitem_41 = None
        mul_100: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_101: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_100, primals_25)
        add_95: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_101, primals_26);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_222: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_95, [4096, 4096])
        addmm_61: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_222, permute_1)
        view_223: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_61, [8, 512, 4096]);  addmm_61 = None
        view_224: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_223, [8, 512, -1, 64]);  view_223 = None
        permute_112: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_62: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_222, permute_3)
        view_226: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_62, [8, 512, 4096]);  addmm_62 = None
        view_227: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_226, [8, 512, -1, 64]);  view_226 = None
        permute_114: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_63: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_222, permute_5)
        view_229: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_63, [8, 512, 4096]);  addmm_63 = None
        view_230: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_229, [8, 512, -1, 64]);  view_229 = None
        permute_116: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_102: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_112, 0.3535533905932738);  permute_112 = None
        permute_117: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        mul_103: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_117, 0.3535533905932738);  permute_117 = None
        expand_43: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_102, [8, 64, 512, 64]);  mul_102 = None
        clone_51: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_231: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_51, [512, 512, 64]);  clone_51 = None
        expand_44: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_103, [8, 64, 64, 512]);  mul_103 = None
        clone_52: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_232: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_52, [512, 64, 512]);  clone_52 = None
        bmm_20: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232)
        view_233: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [8, 64, 512, 512]);  bmm_20 = None
        add_96: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_233, where);  view_233 = None
        amax_10: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_31: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_96, amax_10);  amax_10 = None
        exp_10: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        eq_10: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_96, -inf);  add_96 = None
        logical_not_20: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_21: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_2, div_10);  logical_not_21 = div_10 = None
        expand_45: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_21, [8, 64, 512, 512])
        view_234: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [512, 512, 512]);  expand_45 = None
        expand_46: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_116, [8, 64, 512, 64]);  permute_116 = None
        clone_53: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_235: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_53, [512, 512, 64]);  clone_53 = None
        bmm_21: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = None
        view_236: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [8, 64, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_54: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_54, [8, 512, -1]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_238: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_237, [4096, 4096]);  view_237 = None
        addmm_64: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_238, permute_9)
        view_239: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_64, [8, 512, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_97: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_95, view_239);  add_95 = view_239 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_98: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_32: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_97, getitem_43);  add_97 = getitem_43 = None
        mul_104: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_105: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_104, primals_19)
        add_99: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_105, primals_20);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_240: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_99, [4096, 4096])
        addmm_65: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_240, permute_10)
        view_241: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_65, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        pow_11: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
        mul_107: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_241, mul_107);  view_241 = mul_107 = None
        mul_108: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_101: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_101);  mul_106 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_242: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_109, [4096, 16384]);  mul_109 = None
        addmm_66: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_242, permute_11)
        view_243: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_66, [8, 512, 4096]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_102: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_243, add_99);  view_243 = add_99 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_103: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_33: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_102, getitem_45);  add_102 = getitem_45 = None
        mul_110: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_111: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_110, primals_25)
        add_104: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_111, primals_26);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_244: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_104, [4096, 4096])
        addmm_67: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_12, view_244, permute_1);  primals_12 = permute_1 = None
        view_245: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_67, [8, 512, 4096]);  addmm_67 = None
        view_246: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_245, [8, 512, -1, 64]);  view_245 = None
        permute_123: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_68: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_14, view_244, permute_3);  primals_14 = permute_3 = None
        view_248: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_68, [8, 512, 4096]);  addmm_68 = None
        view_249: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_248, [8, 512, -1, 64]);  view_248 = None
        permute_125: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        addmm_69: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_16, view_244, permute_5);  primals_16 = permute_5 = None
        view_251: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_69, [8, 512, 4096]);  addmm_69 = None
        view_252: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_251, [8, 512, -1, 64]);  view_251 = None
        permute_127: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_112: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_123, 0.3535533905932738);  permute_123 = None
        permute_128: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        mul_113: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_128, 0.3535533905932738);  permute_128 = None
        expand_47: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_112, [8, 64, 512, 64]);  mul_112 = None
        clone_56: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_253: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_56, [512, 512, 64]);  clone_56 = None
        expand_48: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_113, [8, 64, 64, 512]);  mul_113 = None
        clone_57: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_254: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_57, [512, 64, 512]);  clone_57 = None
        bmm_22: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254)
        view_255: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [8, 64, 512, 512]);  bmm_22 = None
        add_105: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_255, where);  view_255 = where = None
        amax_11: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_105, [-1], True)
        sub_34: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_105, amax_11);  amax_11 = None
        exp_11: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        eq_11: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_105, -inf);  add_105 = None
        logical_not_22: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_23: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_2, div_11);  logical_not_23 = full_default_2 = div_11 = None
        expand_49: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_23, [8, 64, 512, 512])
        view_256: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [512, 512, 512]);  expand_49 = None
        expand_50: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_127, [8, 64, 512, 64]);  permute_127 = None
        clone_58: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_257: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_58, [512, 512, 64]);  clone_58 = None
        bmm_23: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = None
        view_258: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 64, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_59: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_259: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_59, [8, 512, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_260: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_259, [4096, 4096]);  view_259 = None
        addmm_70: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_18, view_260, permute_9);  primals_18 = permute_9 = None
        view_261: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_70, [8, 512, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_106: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_104, view_261);  add_104 = view_261 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_107: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_35: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_106, getitem_47);  add_106 = getitem_47 = None
        mul_114: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_115: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_114, primals_19)
        add_108: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_115, primals_20);  mul_115 = primals_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_262: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_108, [4096, 4096])
        addmm_71: "f32[4096, 16384]" = torch.ops.aten.addmm.default(primals_22, view_262, permute_10);  primals_22 = permute_10 = None
        view_263: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_71, [8, 512, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
        pow_12: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
        mul_117: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_263, mul_117);  view_263 = mul_117 = None
        mul_118: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_110: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_110);  mul_116 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_264: "f32[4096, 16384]" = torch.ops.aten.reshape.default(mul_119, [4096, 16384]);  mul_119 = None
        addmm_72: "f32[4096, 4096]" = torch.ops.aten.addmm.default(primals_24, view_264, permute_11);  primals_24 = permute_11 = None
        view_265: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_72, [8, 512, 4096]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_111: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_265, add_108);  view_265 = add_108 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[8, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_112: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_36: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_111, getitem_49);  add_111 = getitem_49 = None
        mul_120: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_121: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_120, primals_25)
        add_113: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_121, primals_26);  mul_121 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        view_266: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_113, [4096, 4096]);  add_113 = None
        permute_133: "f32[4096, 128]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_73: "f32[4096, 128]" = torch.ops.aten.addmm.default(primals_28, view_266, permute_133);  primals_28 = permute_133 = None
        view_267: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [8, 512, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_122: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        pow_13: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 3.0)
        mul_123: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_114: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view_267, mul_123);  view_267 = mul_123 = None
        mul_124: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_114, 0.7978845608028654);  add_114 = None
        tanh_12: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_124);  mul_124 = None
        add_115: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_125: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_122, add_115);  mul_122 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_125, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[8, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_116: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_37: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_125, getitem_51);  mul_125 = None
        mul_126: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_127: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_126, primals_29);  mul_126 = None
        add_117: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_127, primals_30);  mul_127 = primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        view_268: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_117, [4096, 128]);  add_117 = None
        permute_134: "f32[128, 30000]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_74: "f32[4096, 30000]" = torch.ops.aten.addmm.default(primals_31, view_268, permute_134);  primals_31 = permute_134 = None
        view_269: "f32[8, 512, 30000]" = torch.ops.aten.reshape.default(addmm_74, [8, 512, 30000]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:650 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_270: "f32[4096, 30000]" = torch.ops.aten.reshape.default(view_269, [-1, 30000])
        view_271: "i64[4096]" = torch.ops.aten.reshape.default(primals_32, [-1])
        amax_12: "f32[4096, 1]" = torch.ops.aten.amax.default(view_270, [1], True)
        sub_38: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = None
        exp_12: "f32[4096, 30000]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_271, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[4096]" = torch.ops.aten.where.self(ne, view_271, full_default_36);  view_271 = full_default_36 = None
        unsqueeze_3: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_3);  sub_39 = unsqueeze_3 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_25: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_157: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
        permute_158: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_159: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_190: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_191: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_192: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_223: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_224: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_225: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_21: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_22: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_256: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_257: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_258: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_23: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_24: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_289: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_290: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_291: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_25: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_26: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_322: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_323: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        permute_324: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_27: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_28: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_355: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_356: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        permute_357: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_29: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_30: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_388: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_389: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_390: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_31: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_32: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_421: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        permute_422: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_423: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_33: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_34: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_454: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_455: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_456: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_35: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_36: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_487: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_488: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_489: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_37: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_38: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_520: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_521: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_522: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_39: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (div_12, view_269, primals_1, primals_2, primals_4, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_32, gather, mul, view, view_2, where_1, view_18, mul_4, view_20, addmm_5, view_22, mul_10, view_24, where_3, view_40, mul_14, view_42, addmm_11, view_44, mul_20, view_46, where_5, view_62, mul_24, view_64, addmm_17, view_66, mul_30, view_68, where_7, view_84, mul_34, view_86, addmm_23, view_88, mul_40, view_90, where_9, view_106, mul_44, view_108, addmm_29, view_110, mul_50, view_112, where_11, view_128, mul_54, view_130, addmm_35, view_132, mul_60, view_134, where_13, view_150, mul_64, view_152, addmm_41, view_154, mul_70, view_156, where_15, view_172, mul_74, view_174, addmm_47, view_176, mul_80, view_178, where_17, view_194, mul_84, view_196, addmm_53, view_198, mul_90, view_200, where_19, view_216, mul_94, view_218, addmm_59, view_220, mul_100, view_222, where_21, view_238, mul_104, view_240, addmm_65, view_242, mul_110, view_244, where_23, view_260, mul_114, view_262, addmm_71, view_264, mul_120, view_266, addmm_73, getitem_51, rsqrt_25, view_268, view_269, amax_12, log, convert_element_type, div_15, div_16, permute_157, permute_158, permute_159, div_17, div_18, permute_190, permute_191, permute_192, div_19, div_20, permute_223, permute_224, permute_225, div_21, div_22, permute_256, permute_257, permute_258, div_23, div_24, permute_289, permute_290, permute_291, div_25, div_26, permute_322, permute_323, permute_324, div_27, div_28, permute_355, permute_356, permute_357, div_29, div_30, permute_388, permute_389, permute_390, div_31, div_32, permute_421, permute_422, permute_423, div_33, div_34, permute_454, permute_455, permute_456, div_35, div_36, permute_487, permute_488, permute_489, div_37, div_38, permute_520, permute_521, permute_522, div_39)
