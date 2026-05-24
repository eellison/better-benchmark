import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 512]", primals_2: "f32[30522, 768]", primals_3: "i64[1, 512]", primals_4: "f32[512, 768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768, 768]", primals_12: "f32[768]", primals_13: "f32[768, 768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[3072, 768]", primals_18: "f32[3072]", primals_19: "f32[768, 3072]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768, 768]", primals_24: "f32[768]", primals_25: "f32[768, 768]", primals_26: "f32[768]", primals_27: "f32[768, 768]", primals_28: "f32[768]", primals_29: "f32[768, 768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[3072, 768]", primals_34: "f32[3072]", primals_35: "f32[768, 3072]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768, 768]", primals_40: "f32[768]", primals_41: "f32[768, 768]", primals_42: "f32[768]", primals_43: "f32[768, 768]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[3072, 768]", primals_50: "f32[3072]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768, 768]", primals_56: "f32[768]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768, 768]", primals_60: "f32[768]", primals_61: "f32[768, 768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[3072, 768]", primals_66: "f32[3072]", primals_67: "f32[768, 3072]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768, 768]", primals_72: "f32[768]", primals_73: "f32[768, 768]", primals_74: "f32[768]", primals_75: "f32[768, 768]", primals_76: "f32[768]", primals_77: "f32[768, 768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[3072, 768]", primals_82: "f32[3072]", primals_83: "f32[768, 3072]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[768, 768]", primals_88: "f32[768]", primals_89: "f32[768, 768]", primals_90: "f32[768]", primals_91: "f32[768, 768]", primals_92: "f32[768]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[3072, 768]", primals_98: "f32[3072]", primals_99: "f32[768, 3072]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768, 768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[30522]", primals_108: "i64[8, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "f32[8, 512, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_3);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = None
        mul: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_2: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:121 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_12: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_2);  add_2 = None
        mul_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[512]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [8, -1, 512, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view: "f32[4096, 768]" = torch.ops.aten.reshape.default(mul_3, [4096, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_8, view, permute);  primals_8 = permute = None
        view_1: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm, [8, 512, 768]);  addmm = None
        view_2: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [8, 512, -1, 64]);  view_1 = None
        permute_1: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_10, view, permute_2);  primals_10 = permute_2 = None
        view_4: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 512, 768]);  addmm_1 = None
        view_5: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [8, 512, -1, 64]);  view_4 = None
        permute_3: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_12, view, permute_4);  primals_12 = permute_4 = None
        view_7: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [8, 512, 768]);  addmm_2 = None
        view_8: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [8, 512, -1, 64]);  view_7 = None
        permute_5: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  expand = full_default = None
        mul_4: "f32[8, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_5: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_1: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(mul_4, [8, 12, 512, 64]);  mul_4 = None
        clone: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone, [96, 512, 64]);  clone = None
        expand_2: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_5, [8, 12, 64, 512]);  mul_5 = None
        clone_1: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_1, [96, 64, 512]);  clone_1 = None
        bmm: "f32[96, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [8, 12, 512, 512])
        add_5: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = None
        amax: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_5, [-1], True)
        sub_1: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_5, amax)
        exp: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        eq: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_5, -inf);  add_5 = None
        logical_not: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[8, 12, 512, 512]" = torch.ops.aten.full.default([8, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  div = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_11: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_6: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_1, where_1);  where_1 = None
        mul_7: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_3: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_7, [8, 12, 512, 512]);  mul_7 = None
        view_12: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_3, [96, 512, 512]);  expand_3 = None
        expand_4: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [8, 12, 512, 64]);  permute_5 = None
        clone_2: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [96, 512, 64]);  clone_2 = None
        bmm_1: "f32[96, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "f32[8, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "f32[8, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [8, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_16: "f32[4096, 768]" = torch.ops.aten.reshape.default(view_15, [4096, 768]);  view_15 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_3: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_14, view_16, permute_8);  primals_14 = permute_8 = None
        view_17: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_6: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(view_17, mul_3);  view_17 = mul_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_2: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  add_6 = getitem_3 = None
        mul_8: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_9: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_15)
        add_8: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_16);  mul_9 = primals_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_18: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_8, [4096, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_4: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_18, view_18, permute_9);  primals_18 = permute_9 = None
        view_19: "f32[8, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_10: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_11: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[8, 512, 3072]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_9: "f32[8, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_12: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_10, add_9);  mul_10 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_20: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_12, [4096, 3072]);  mul_12 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        addmm_5: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_20, view_20, permute_10);  primals_20 = permute_10 = None
        view_21: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_10: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_13: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_21);  view_21 = None
        mul_14: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, 1.1111111111111112);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_10: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_14, add_8);  mul_14 = add_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
        mul_15: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_16: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_21)
        add_12: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_22);  mul_16 = primals_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_22: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_12, [4096, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_6: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_24, view_22, permute_11);  primals_24 = permute_11 = None
        view_23: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [8, 512, 768]);  addmm_6 = None
        view_24: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [8, 512, -1, 64]);  view_23 = None
        permute_12: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_7: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_26, view_22, permute_13);  primals_26 = permute_13 = None
        view_26: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 512, 768]);  addmm_7 = None
        view_27: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_26, [8, 512, -1, 64]);  view_26 = None
        permute_14: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_8: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_28, view_22, permute_15);  primals_28 = permute_15 = None
        view_29: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [8, 512, 768]);  addmm_8 = None
        view_30: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_29, [8, 512, -1, 64]);  view_29 = None
        permute_16: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_17: "f32[8, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        permute_17: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_18: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_5: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(mul_17, [8, 12, 512, 64]);  mul_17 = None
        clone_4: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_31: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_4, [96, 512, 64]);  clone_4 = None
        expand_6: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_18, [8, 12, 64, 512]);  mul_18 = None
        clone_5: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_32: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_5, [96, 64, 512]);  clone_5 = None
        bmm_2: "f32[96, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [8, 12, 512, 512]);  bmm_2 = None
        add_13: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_33, where);  view_33 = None
        amax_1: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_4: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_13, amax_1);  amax_1 = None
        exp_1: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        eq_1: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_13, -inf);  add_13 = None
        logical_not_2: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_2, div_1);  logical_not_3 = div_1 = None
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_9: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_19: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_3, where_3)
        mul_20: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_19, 1.1111111111111112);  mul_19 = None
        expand_7: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_20, [8, 12, 512, 512]);  mul_20 = None
        view_34: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_7, [96, 512, 512]);  expand_7 = None
        expand_8: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, [8, 12, 512, 64]);  permute_16 = None
        clone_6: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_6, [96, 512, 64]);  clone_6 = None
        bmm_3: "f32[96, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "f32[8, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 12, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "f32[8, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [8, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_38: "f32[4096, 768]" = torch.ops.aten.reshape.default(view_37, [4096, 768]);  view_37 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_9: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_30, view_38, permute_19);  primals_30 = permute_19 = None
        view_39: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_14: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(view_39, add_12);  view_39 = add_12 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_15: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
        mul_21: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_22: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_31)
        add_16: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_32);  mul_22 = primals_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_40: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_16, [4096, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_33, [1, 0])
        addmm_10: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_34, view_40, permute_20);  primals_34 = permute_20 = None
        view_41: "f32[8, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_23: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_24: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[8, 512, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_17: "f32[8, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_25: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_23, add_17);  mul_23 = add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_42: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_25, [4096, 3072]);  mul_25 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_11: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_36, view_42, permute_21);  primals_36 = permute_21 = None
        view_43: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_8: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_26: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_4, view_43);  view_43 = None
        mul_27: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_18: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, add_16);  mul_27 = add_16 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_19: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
        mul_28: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_29: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_37)
        add_20: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_38);  mul_29 = primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_44: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_20, [4096, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0])
        addmm_12: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_40, view_44, permute_22);  primals_40 = permute_22 = None
        view_45: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [8, 512, 768]);  addmm_12 = None
        view_46: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [8, 512, -1, 64]);  view_45 = None
        permute_23: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_13: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_42, view_44, permute_24);  primals_42 = permute_24 = None
        view_48: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 512, 768]);  addmm_13 = None
        view_49: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_48, [8, 512, -1, 64]);  view_48 = None
        permute_25: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        addmm_14: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_44, view_44, permute_26);  primals_44 = permute_26 = None
        view_51: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [8, 512, 768]);  addmm_14 = None
        view_52: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_51, [8, 512, -1, 64]);  view_51 = None
        permute_27: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_30: "f32[8, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        permute_28: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_31: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_9: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(mul_30, [8, 12, 512, 64]);  mul_30 = None
        clone_8: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_53: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [96, 512, 64]);  clone_8 = None
        expand_10: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_31, [8, 12, 64, 512]);  mul_31 = None
        clone_9: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_54: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_9, [96, 64, 512]);  clone_9 = None
        bmm_4: "f32[96, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [8, 12, 512, 512]);  bmm_4 = None
        add_21: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_55, where);  view_55 = None
        amax_2: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_21, [-1], True)
        sub_7: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_21, amax_2);  amax_2 = None
        exp_2: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        eq_2: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_21, -inf);  add_21 = None
        logical_not_4: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_2, div_2);  logical_not_5 = div_2 = None
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_7: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_32: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_5, where_5)
        mul_33: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None
        expand_11: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_33, [8, 12, 512, 512]);  mul_33 = None
        view_56: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_11, [96, 512, 512]);  expand_11 = None
        expand_12: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [8, 12, 512, 64]);  permute_27 = None
        clone_10: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_57: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [96, 512, 64]);  clone_10 = None
        bmm_5: "f32[96, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "f32[8, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 12, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "f32[8, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [8, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_60: "f32[4096, 768]" = torch.ops.aten.reshape.default(view_59, [4096, 768]);  view_59 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_15: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_46, view_60, permute_30);  primals_46 = permute_30 = None
        view_61: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_22: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(view_61, add_20);  view_61 = add_20 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_8: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_11);  add_22 = getitem_11 = None
        mul_34: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_35: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_47)
        add_24: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_48);  mul_35 = primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_62: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_24, [4096, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0])
        addmm_16: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_50, view_62, permute_31);  primals_50 = permute_31 = None
        view_63: "f32[8, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_36: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_37: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[8, 512, 3072]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
        add_25: "f32[8, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_38: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_25);  mul_36 = add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_64: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_38, [4096, 3072]);  mul_38 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_17: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_52, view_64, permute_32);  primals_52 = permute_32 = None
        view_65: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_6: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_39: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_65);  view_65 = None
        mul_40: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_39, 1.1111111111111112);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_26: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_40, add_24);  mul_40 = add_24 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_27: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_9: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_26, getitem_13);  add_26 = getitem_13 = None
        mul_41: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_42: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_53)
        add_28: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_54);  mul_42 = primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_66: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_28, [4096, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        addmm_18: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_56, view_66, permute_33);  primals_56 = permute_33 = None
        view_67: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [8, 512, 768]);  addmm_18 = None
        view_68: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [8, 512, -1, 64]);  view_67 = None
        permute_34: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_19: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_58, view_66, permute_35);  primals_58 = permute_35 = None
        view_70: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 512, 768]);  addmm_19 = None
        view_71: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_70, [8, 512, -1, 64]);  view_70 = None
        permute_36: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_20: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_60, view_66, permute_37);  primals_60 = permute_37 = None
        view_73: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [8, 512, 768]);  addmm_20 = None
        view_74: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_73, [8, 512, -1, 64]);  view_73 = None
        permute_38: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_43: "f32[8, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        permute_39: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_44: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_13: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(mul_43, [8, 12, 512, 64]);  mul_43 = None
        clone_12: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_75: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_12, [96, 512, 64]);  clone_12 = None
        expand_14: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_44, [8, 12, 64, 512]);  mul_44 = None
        clone_13: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_76: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_13, [96, 64, 512]);  clone_13 = None
        bmm_6: "f32[96, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [8, 12, 512, 512]);  bmm_6 = None
        add_29: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_77, where);  view_77 = None
        amax_3: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_29, [-1], True)
        sub_10: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_29, amax_3);  amax_3 = None
        exp_3: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        eq_3: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_29, -inf);  add_29 = None
        logical_not_6: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_2, div_3);  logical_not_7 = div_3 = None
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_5: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_45: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_7, where_7)
        mul_46: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None
        expand_15: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_46, [8, 12, 512, 512]);  mul_46 = None
        view_78: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_15, [96, 512, 512]);  expand_15 = None
        expand_16: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, [8, 12, 512, 64]);  permute_38 = None
        clone_14: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_79: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_14, [96, 512, 64]);  clone_14 = None
        bmm_7: "f32[96, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
        view_80: "f32[8, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 12, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "f32[8, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(clone_15, [8, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_82: "f32[4096, 768]" = torch.ops.aten.reshape.default(view_81, [4096, 768]);  view_81 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        addmm_21: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_62, view_82, permute_41);  primals_62 = permute_41 = None
        view_83: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_30: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(view_83, add_28);  view_83 = add_28 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_31: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_11: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_15);  add_30 = getitem_15 = None
        mul_47: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_48: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_47, primals_63)
        add_32: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_48, primals_64);  mul_48 = primals_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_84: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_32, [4096, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_65, [1, 0])
        addmm_22: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_66, view_84, permute_42);  primals_66 = permute_42 = None
        view_85: "f32[8, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_49: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_50: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[8, 512, 3072]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_33: "f32[8, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_51: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_49, add_33);  mul_49 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_86: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_51, [4096, 3072]);  mul_51 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        addmm_23: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_68, view_86, permute_43);  primals_68 = permute_43 = None
        view_87: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_4: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_52: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_87);  view_87 = None
        mul_53: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, 1.1111111111111112);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_34: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, add_32);  mul_53 = add_32 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_35: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_12: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_34, getitem_17);  add_34 = getitem_17 = None
        mul_54: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_55: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_69)
        add_36: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_70);  mul_55 = primals_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_88: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_36, [4096, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0])
        addmm_24: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_72, view_88, permute_44);  primals_72 = permute_44 = None
        view_89: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [8, 512, 768]);  addmm_24 = None
        view_90: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [8, 512, -1, 64]);  view_89 = None
        permute_45: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0])
        addmm_25: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_74, view_88, permute_46);  primals_74 = permute_46 = None
        view_92: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 512, 768]);  addmm_25 = None
        view_93: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_92, [8, 512, -1, 64]);  view_92 = None
        permute_47: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_26: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_76, view_88, permute_48);  primals_76 = permute_48 = None
        view_95: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [8, 512, 768]);  addmm_26 = None
        view_96: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_95, [8, 512, -1, 64]);  view_95 = None
        permute_49: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_56: "f32[8, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        permute_50: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_57: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_17: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(mul_56, [8, 12, 512, 64]);  mul_56 = None
        clone_16: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_97: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [96, 512, 64]);  clone_16 = None
        expand_18: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_57, [8, 12, 64, 512]);  mul_57 = None
        clone_17: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_98: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_17, [96, 64, 512]);  clone_17 = None
        bmm_8: "f32[96, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [8, 12, 512, 512]);  bmm_8 = None
        add_37: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_99, where);  view_99 = None
        amax_4: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_37, [-1], True)
        sub_13: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_37, amax_4);  amax_4 = None
        exp_4: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        eq_4: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_37, -inf);  add_37 = None
        logical_not_8: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_2, div_4);  logical_not_9 = div_4 = None
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_3: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_58: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_9, where_9)
        mul_59: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        expand_19: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_59, [8, 12, 512, 512]);  mul_59 = None
        view_100: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_19, [96, 512, 512]);  expand_19 = None
        expand_20: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, [8, 12, 512, 64]);  permute_49 = None
        clone_18: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_101: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [96, 512, 64]);  clone_18 = None
        bmm_9: "f32[96, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
        view_102: "f32[8, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 12, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "f32[8, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [8, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_104: "f32[4096, 768]" = torch.ops.aten.reshape.default(view_103, [4096, 768]);  view_103 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_27: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_78, view_104, permute_52);  primals_78 = permute_52 = None
        view_105: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_38: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(view_105, add_36);  view_105 = add_36 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_39: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_14: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_19);  add_38 = getitem_19 = None
        mul_60: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_61: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_79)
        add_40: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_80);  mul_61 = primals_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_106: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_40, [4096, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0])
        addmm_28: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_82, view_106, permute_53);  primals_82 = permute_53 = None
        view_107: "f32[8, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_62: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_63: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[8, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_41: "f32[8, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_64: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_41);  mul_62 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_108: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_64, [4096, 3072]);  mul_64 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_29: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_84, view_108, permute_54);  primals_84 = permute_54 = None
        view_109: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_2: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_65: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_10, view_109);  view_109 = None
        mul_66: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_42: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, add_40);  mul_66 = add_40 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_43: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_15: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_21);  add_42 = getitem_21 = None
        mul_67: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_68: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, primals_85)
        add_44: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, primals_86);  mul_68 = primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_110: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_44, [4096, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        addmm_30: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_88, view_110, permute_55);  primals_88 = permute_55 = None
        view_111: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [8, 512, 768]);  addmm_30 = None
        view_112: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [8, 512, -1, 64]);  view_111 = None
        permute_56: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_31: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_90, view_110, permute_57);  primals_90 = permute_57 = None
        view_114: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 512, 768]);  addmm_31 = None
        view_115: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_114, [8, 512, -1, 64]);  view_114 = None
        permute_58: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_32: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_92, view_110, permute_59);  primals_92 = permute_59 = None
        view_117: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [8, 512, 768]);  addmm_32 = None
        view_118: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(view_117, [8, 512, -1, 64]);  view_117 = None
        permute_60: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_69: "f32[8, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        permute_61: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_70: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_21: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(mul_69, [8, 12, 512, 64]);  mul_69 = None
        clone_20: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_119: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_20, [96, 512, 64]);  clone_20 = None
        expand_22: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_70, [8, 12, 64, 512]);  mul_70 = None
        clone_21: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_120: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_21, [96, 64, 512]);  clone_21 = None
        bmm_10: "f32[96, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [8, 12, 512, 512]);  bmm_10 = None
        add_45: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_121, where);  view_121 = where = None
        amax_5: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_45, [-1], True)
        sub_16: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_45, amax_5);  amax_5 = None
        exp_5: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        eq_5: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_45, -inf);  add_45 = None
        logical_not_10: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_2, div_5);  logical_not_11 = full_default_2 = div_5 = None
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_1: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_71: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_11, where_11)
        mul_72: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None
        expand_23: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_72, [8, 12, 512, 512]);  mul_72 = None
        view_122: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_23, [96, 512, 512]);  expand_23 = None
        expand_24: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, [8, 12, 512, 64]);  permute_60 = None
        clone_22: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_123: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_22, [96, 512, 64]);  clone_22 = None
        bmm_11: "f32[96, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "f32[8, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 12, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "f32[8, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(clone_23, [8, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_126: "f32[4096, 768]" = torch.ops.aten.reshape.default(view_125, [4096, 768]);  view_125 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_33: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_94, view_126, permute_63);  primals_94 = permute_63 = None
        view_127: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_46: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(view_127, add_44);  view_127 = add_44 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_47: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_17: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_23);  add_46 = getitem_23 = None
        mul_73: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_74: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_95)
        add_48: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_96);  mul_74 = primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_128: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_48, [4096, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0])
        addmm_34: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_98, view_128, permute_64);  primals_98 = permute_64 = None
        view_129: "f32[8, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_75: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_76: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[8, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_49: "f32[8, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_77: "f32[8, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_49);  mul_75 = add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_130: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_77, [4096, 3072]);  mul_77 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_35: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_100, view_130, permute_65);  primals_100 = permute_65 = None
        view_131: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_78: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_131);  view_131 = None
        mul_79: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_50: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, add_48);  mul_79 = add_48 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_51: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_18: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_25);  add_50 = getitem_25 = None
        mul_80: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_81: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_101)
        add_52: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_102);  mul_81 = primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_132: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_52, [4096, 768]);  add_52 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0])
        addmm_36: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_104, view_132, permute_66);  primals_104 = permute_66 = None
        view_133: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [8, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_82: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_133, 0.5)
        mul_83: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476);  view_133 = None
        erf_6: "f32[8, 512, 768]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_53: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_84: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, add_53);  mul_82 = add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        var_mean_13 = torch.ops.aten.var_mean.correction(mul_84, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_54: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_19: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_84, getitem_27);  mul_84 = None
        mul_85: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
        mul_86: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_105);  mul_85 = None
        add_55: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_106);  mul_86 = primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_134: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_55, [4096, 768]);  add_55 = None
        permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm_37: "f32[4096, 30522]" = torch.ops.aten.addmm.default(primals_107, view_134, permute_67);  primals_107 = permute_67 = None
        view_135: "f32[8, 512, 30522]" = torch.ops.aten.reshape.default(addmm_37, [8, 512, 30522]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_136: "f32[4096, 30522]" = torch.ops.aten.reshape.default(view_135, [-1, 30522])
        view_137: "i64[4096]" = torch.ops.aten.reshape.default(primals_108, [-1])
        amax_6: "f32[4096, 1]" = torch.ops.aten.amax.default(view_136, [1], True)
        sub_20: "f32[4096, 30522]" = torch.ops.aten.sub.Tensor(view_136, amax_6);  view_136 = None
        exp_6: "f32[4096, 30522]" = torch.ops.aten.exp.default(sub_20)
        sum_7: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_21: "f32[4096, 30522]" = torch.ops.aten.sub.Tensor(sub_20, log);  sub_20 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_137, -100)
        full_default_18: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[4096]" = torch.ops.aten.where.self(ne, view_137, full_default_18);  view_137 = full_default_18 = None
        unsqueeze_3: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_21, 1, unsqueeze_3);  sub_21 = unsqueeze_3 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_13: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_8: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        div_6: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_9: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_10: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_89: "f32[96, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_90: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_91: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_92: "f32[96, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_11: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_12: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_122: "f32[96, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_123: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_124: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_125: "f32[96, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_13: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_155: "f32[96, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_156: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_157: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_158: "f32[96, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_188: "f32[96, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_189: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_190: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_191: "f32[96, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_221: "f32[96, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_222: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_223: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_224: "f32[96, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_254: "f32[96, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_255: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_256: "f32[96, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_257: "f32[96, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (div_6, view_135, primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, embedding, embedding_1, getitem_1, rsqrt, gt, ge, view, bmm, amax, sum_1, logical_not_1, gt_1, view_16, mul_8, view_18, addmm_4, view_20, gt_2, mul_15, view_22, where_3, gt_3, view_38, mul_21, view_40, addmm_10, view_42, gt_4, mul_28, view_44, where_5, gt_5, view_60, mul_34, view_62, addmm_16, view_64, gt_6, mul_41, view_66, where_7, gt_7, view_82, mul_47, view_84, addmm_22, view_86, gt_8, mul_54, view_88, where_9, gt_9, view_104, mul_60, view_106, addmm_28, view_108, gt_10, mul_67, view_110, where_11, gt_11, view_126, mul_73, view_128, addmm_34, view_130, gt_12, mul_80, view_132, addmm_36, getitem_27, rsqrt_13, view_134, view_135, amax_6, log, convert_element_type, div_9, div_10, permute_89, permute_90, permute_91, permute_92, div_11, div_12, permute_122, permute_123, permute_124, permute_125, div_13, div_14, permute_155, permute_156, permute_157, permute_158, div_15, div_16, permute_188, permute_189, permute_190, permute_191, div_17, div_18, permute_221, permute_222, permute_223, permute_224, div_19, div_20, permute_254, permute_255, permute_256, permute_257)
