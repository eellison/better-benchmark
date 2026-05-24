import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[64, 512]", primals_2: "i64[64, 512]", primals_3: "i64[1, 512]", primals_4: "i64[1, 512]", primals_5: "f32[30522, 128]", primals_6: "f32[2, 128]", primals_7: "f32[512, 128]", primals_8: "f32[128]", primals_9: "f32[128]", primals_10: "f32[256, 128]", primals_11: "f32[256]", primals_12: "f32[256, 256]", primals_13: "f32[256]", primals_14: "f32[256, 256]", primals_15: "f32[256]", primals_16: "f32[256, 256]", primals_17: "f32[256]", primals_18: "f32[256, 256]", primals_19: "f32[256]", primals_20: "f32[256]", primals_21: "f32[256]", primals_22: "f32[1024, 256]", primals_23: "f32[1024]", primals_24: "f32[256, 1024]", primals_25: "f32[256]", primals_26: "f32[256]", primals_27: "f32[256]", primals_28: "f32[256, 256]", primals_29: "f32[256]", primals_30: "f32[256, 256]", primals_31: "f32[256]", primals_32: "f32[256, 256]", primals_33: "f32[256]", primals_34: "f32[256, 256]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[256]", primals_38: "f32[1024, 256]", primals_39: "f32[1024]", primals_40: "f32[256, 1024]", primals_41: "f32[256]", primals_42: "f32[256]", primals_43: "f32[256]", primals_44: "f32[256, 256]", primals_45: "f32[256]", primals_46: "f32[256, 256]", primals_47: "f32[256]", primals_48: "f32[256, 256]", primals_49: "f32[256]", primals_50: "f32[256, 256]", primals_51: "f32[256]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[1024, 256]", primals_55: "f32[1024]", primals_56: "f32[256, 1024]", primals_57: "f32[256]", primals_58: "f32[256]", primals_59: "f32[256]", primals_60: "f32[256, 256]", primals_61: "f32[256]", primals_62: "f32[256, 256]", primals_63: "f32[256]", primals_64: "f32[256, 256]", primals_65: "f32[256]", primals_66: "f32[256, 256]", primals_67: "f32[256]", primals_68: "f32[256]", primals_69: "f32[256]", primals_70: "f32[1024, 256]", primals_71: "f32[1024]", primals_72: "f32[256, 1024]", primals_73: "f32[256]", primals_74: "f32[256]", primals_75: "f32[256]", primals_76: "f32[256, 256]", primals_77: "f32[256]", primals_78: "f32[256, 256]", primals_79: "f32[256]", primals_80: "f32[256, 256]", primals_81: "f32[256]", primals_82: "f32[256, 256]", primals_83: "f32[256]", primals_84: "f32[256]", primals_85: "f32[256]", primals_86: "f32[1024, 256]", primals_87: "f32[1024]", primals_88: "f32[256, 1024]", primals_89: "f32[256]", primals_90: "f32[256]", primals_91: "f32[256]", primals_92: "f32[256, 256]", primals_93: "f32[256]", primals_94: "f32[256, 256]", primals_95: "f32[256]", primals_96: "f32[256, 256]", primals_97: "f32[256]", primals_98: "f32[256, 256]", primals_99: "f32[256]", primals_100: "f32[256]", primals_101: "f32[256]", primals_102: "f32[1024, 256]", primals_103: "f32[1024]", primals_104: "f32[256, 1024]", primals_105: "f32[256]", primals_106: "f32[256]", primals_107: "f32[256]", primals_108: "f32[256, 256]", primals_109: "f32[256]", primals_110: "f32[256, 256]", primals_111: "f32[256]", primals_112: "f32[256, 256]", primals_113: "f32[256]", primals_114: "f32[256, 256]", primals_115: "f32[256]", primals_116: "f32[256]", primals_117: "f32[256]", primals_118: "f32[1024, 256]", primals_119: "f32[1024]", primals_120: "f32[256, 1024]", primals_121: "f32[256]", primals_122: "f32[256]", primals_123: "f32[256]", primals_124: "f32[256, 256]", primals_125: "f32[256]", primals_126: "f32[256, 256]", primals_127: "f32[256]", primals_128: "f32[256, 256]", primals_129: "f32[256]", primals_130: "f32[256, 256]", primals_131: "f32[256]", primals_132: "f32[256]", primals_133: "f32[256]", primals_134: "f32[1024, 256]", primals_135: "f32[1024]", primals_136: "f32[256, 1024]", primals_137: "f32[256]", primals_138: "f32[256]", primals_139: "f32[256]", primals_140: "f32[256, 256]", primals_141: "f32[256]", primals_142: "f32[256, 256]", primals_143: "f32[256]", primals_144: "f32[256, 256]", primals_145: "f32[256]", primals_146: "f32[256, 256]", primals_147: "f32[256]", primals_148: "f32[256]", primals_149: "f32[256]", primals_150: "f32[1024, 256]", primals_151: "f32[1024]", primals_152: "f32[256, 1024]", primals_153: "f32[256]", primals_154: "f32[256]", primals_155: "f32[256]", primals_156: "f32[256, 256]", primals_157: "f32[256]", primals_158: "f32[256, 256]", primals_159: "f32[256]", primals_160: "f32[256, 256]", primals_161: "f32[256]", primals_162: "f32[256, 256]", primals_163: "f32[256]", primals_164: "f32[256]", primals_165: "f32[256]", primals_166: "f32[1024, 256]", primals_167: "f32[1024]", primals_168: "f32[256, 1024]", primals_169: "f32[256]", primals_170: "f32[256]", primals_171: "f32[256]", primals_172: "f32[256, 256]", primals_173: "f32[256]", primals_174: "f32[256, 256]", primals_175: "f32[256]", primals_176: "f32[256, 256]", primals_177: "f32[256]", primals_178: "f32[256, 256]", primals_179: "f32[256]", primals_180: "f32[256]", primals_181: "f32[256]", primals_182: "f32[1024, 256]", primals_183: "f32[1024]", primals_184: "f32[256, 1024]", primals_185: "f32[256]", primals_186: "f32[256]", primals_187: "f32[256]", primals_188: "f32[256, 256]", primals_189: "f32[256]", primals_190: "f32[256, 256]", primals_191: "f32[256]", primals_192: "f32[256, 256]", primals_193: "f32[256]", primals_194: "f32[256, 256]", primals_195: "f32[256]", primals_196: "f32[256]", primals_197: "f32[256]", primals_198: "f32[1024, 256]", primals_199: "f32[1024]", primals_200: "f32[256, 1024]", primals_201: "f32[256]", primals_202: "f32[256]", primals_203: "f32[256]", primals_204: "f32[128, 256]", primals_205: "f32[128]", primals_206: "f32[128]", primals_207: "f32[128]", primals_208: "f32[30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:101 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(primals_4, [1, -1]);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:102 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, primals_3);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[64, 512]" = torch.ops.aten.expand.default(gather, [64, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(primals_5, primals_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(primals_6, expand_1);  primals_6 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:110 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_7, primals_3);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean[0]
        getitem_1: "f32[64, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul, primals_8)
        add_3: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_9);  mul_1 = primals_9 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:116 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_36: "f32[64, 512, 128]" = torch.ops.prims.inductor_random.default([64, 512, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[64, 512, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        view: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_3, [32768, 128]);  mul_3 = None
        permute: "f32[128, 256]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_11, view, permute);  primals_11 = permute = None
        view_1: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm, [64, 512, 256]);  addmm = None

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
        expand_2: "b8[64, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [64, -1, 512, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_1, [32768, 256])
        permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_1: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_13, view_2, permute_1);  primals_13 = permute_1 = None
        view_3: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_1, [64, 512, 256]);  addmm_1 = None
        view_4: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_3, [64, 512, -1, 64]);  view_3 = None
        permute_2: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_2: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_15, view_2, permute_3);  primals_15 = permute_3 = None
        view_6: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_2, [64, 512, 256]);  addmm_2 = None
        view_7: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_6, [64, 512, -1, 64]);  view_6 = None
        permute_4: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_5: "f32[256, 256]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_3: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_17, view_2, permute_5);  primals_17 = permute_5 = None
        view_9: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_3, [64, 512, 256]);  addmm_3 = None
        view_10: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_9, [64, 512, -1, 64]);  view_9 = None
        permute_6: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  expand_2 = full_default = None
        mul_4: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_2, 0.3535533905932738);  permute_2 = None
        permute_7: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_5: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_7, 0.3535533905932738);  permute_7 = None
        expand_3: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_4, [64, 4, 512, 64]);  mul_4 = None
        clone: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_11: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone, [256, 512, 64]);  clone = None
        expand_4: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_5, [64, 4, 64, 512]);  mul_5 = None
        clone_1: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_12: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_1, [256, 64, 512]);  clone_1 = None
        bmm: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12)
        view_13: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm, [64, 4, 512, 512])
        add_6: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = None
        amax: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax)
        exp: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        eq: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_6, -inf);  add_6 = None
        logical_not: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  div = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_35: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_6: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_1, where_1);  where_1 = None
        mul_7: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_5: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_7, [64, 4, 512, 512]);  mul_7 = None
        view_14: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [256, 512, 512]);  expand_5 = None
        expand_6: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_6, [64, 4, 512, 64]);  permute_6 = None
        clone_2: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_15: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [256, 512, 64]);  clone_2 = None
        bmm_1: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15)
        view_16: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [64, 4, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_3: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_17: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_3, [64, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_17, [32768, 256]);  view_17 = None
        permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_4: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_19, view_18, permute_9);  primals_19 = permute_9 = None
        view_19: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_4, [64, 512, 256]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_34: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_8: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_2, view_19);  view_19 = None
        mul_9: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_9, view_1);  mul_9 = view_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_2: "f32[64, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[64, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_8: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_7, getitem_3);  add_7 = getitem_3 = None
        mul_10: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, primals_20)
        add_9: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_11, primals_21);  mul_11 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_9, [32768, 256])
        permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        addmm_5: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_23, view_20, permute_10);  primals_23 = permute_10 = None
        view_21: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_5, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_12: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_13: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_12, add_10);  mul_12 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_22: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_14, [32768, 1024]);  mul_14 = None
        permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_6: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_25, view_22, permute_11);  primals_25 = permute_11 = None
        view_23: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_6, [64, 512, 256]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_33: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_15: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_3, view_23);  view_23 = None
        mul_16: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_16, add_9);  mul_16 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[64, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[64, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        mul_17: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_18: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, primals_26)
        add_13: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_18, primals_27);  mul_18 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_24: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_13, [32768, 256])
        permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_7: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_29, view_24, permute_12);  primals_29 = permute_12 = None
        view_25: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_7, [64, 512, 256]);  addmm_7 = None
        view_26: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_25, [64, 512, -1, 64]);  view_25 = None
        permute_13: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_14: "f32[256, 256]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_8: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_31, view_24, permute_14);  primals_31 = permute_14 = None
        view_28: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_8, [64, 512, 256]);  addmm_8 = None
        view_29: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_28, [64, 512, -1, 64]);  view_28 = None
        permute_15: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_16: "f32[256, 256]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_9: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_33, view_24, permute_16);  primals_33 = permute_16 = None
        view_31: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_9, [64, 512, 256]);  addmm_9 = None
        view_32: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_31, [64, 512, -1, 64]);  view_31 = None
        permute_17: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_19: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_13, 0.3535533905932738);  permute_13 = None
        permute_18: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_20: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_18, 0.3535533905932738);  permute_18 = None
        expand_7: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_19, [64, 4, 512, 64]);  mul_19 = None
        clone_4: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_33: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_4, [256, 512, 64]);  clone_4 = None
        expand_8: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_20, [64, 4, 64, 512]);  mul_20 = None
        clone_5: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_34: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_5, [256, 64, 512]);  clone_5 = None
        bmm_2: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34)
        view_35: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [64, 4, 512, 512]);  bmm_2 = None
        add_14: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_35, where);  view_35 = None
        amax_1: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
        sub_4: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  amax_1 = None
        exp_1: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        eq_1: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_14, -inf);  add_14 = None
        logical_not_2: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_2, div_1);  logical_not_3 = div_1 = None
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_32: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_21: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_4, where_3)
        mul_22: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_21, 1.1111111111111112);  mul_21 = None
        expand_9: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_22, [64, 4, 512, 512]);  mul_22 = None
        view_36: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [256, 512, 512]);  expand_9 = None
        expand_10: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_17, [64, 4, 512, 64]);  permute_17 = None
        clone_6: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_37: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_6, [256, 512, 64]);  clone_6 = None
        bmm_3: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37)
        view_38: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [64, 4, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_7: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_39: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_7, [64, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_39, [32768, 256]);  view_39 = None
        permute_20: "f32[256, 256]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        addmm_10: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_35, view_40, permute_20);  primals_35 = permute_20 = None
        view_41: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_10, [64, 512, 256]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_31: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_23: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_5, view_41);  view_41 = None
        mul_24: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_24, add_13);  mul_24 = add_13 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[64, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        mul_25: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_26: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_25, primals_36)
        add_17: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_26, primals_37);  mul_26 = primals_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_17, [32768, 256])
        permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        addmm_11: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_39, view_42, permute_21);  primals_39 = permute_21 = None
        view_43: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_11, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_27: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_28: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_29: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_27, add_18);  mul_27 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_29, [32768, 1024]);  mul_29 = None
        permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        addmm_12: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_41, view_44, permute_22);  primals_41 = permute_22 = None
        view_45: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_12, [64, 512, 256]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_30: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_30: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_6, view_45);  view_45 = None
        mul_31: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_31, add_17);  mul_31 = add_17 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[64, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_32: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_33: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_32, primals_42)
        add_21: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_33, primals_43);  mul_33 = primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_46: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_21, [32768, 256])
        permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_13: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_45, view_46, permute_23);  primals_45 = permute_23 = None
        view_47: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_13, [64, 512, 256]);  addmm_13 = None
        view_48: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_47, [64, 512, -1, 64]);  view_47 = None
        permute_24: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_25: "f32[256, 256]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_14: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_47, view_46, permute_25);  primals_47 = permute_25 = None
        view_50: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_14, [64, 512, 256]);  addmm_14 = None
        view_51: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_50, [64, 512, -1, 64]);  view_50 = None
        permute_26: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_27: "f32[256, 256]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_15: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_49, view_46, permute_27);  primals_49 = permute_27 = None
        view_53: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_15, [64, 512, 256]);  addmm_15 = None
        view_54: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_53, [64, 512, -1, 64]);  view_53 = None
        permute_28: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_34: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_24, 0.3535533905932738);  permute_24 = None
        permute_29: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_35: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_29, 0.3535533905932738);  permute_29 = None
        expand_11: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_34, [64, 4, 512, 64]);  mul_34 = None
        clone_8: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_55: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [256, 512, 64]);  clone_8 = None
        expand_12: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_35, [64, 4, 64, 512]);  mul_35 = None
        clone_9: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_56: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_9, [256, 64, 512]);  clone_9 = None
        bmm_4: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56)
        view_57: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [64, 4, 512, 512]);  bmm_4 = None
        add_22: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_57, where);  view_57 = None
        amax_2: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_7: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  amax_2 = None
        exp_2: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        eq_2: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_22, -inf);  add_22 = None
        logical_not_4: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_2, div_2);  logical_not_5 = div_2 = None
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_29: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_36: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_7, where_5)
        mul_37: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        expand_13: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_37, [64, 4, 512, 512]);  mul_37 = None
        view_58: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [256, 512, 512]);  expand_13 = None
        expand_14: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_28, [64, 4, 512, 64]);  permute_28 = None
        clone_10: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_59: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [256, 512, 64]);  clone_10 = None
        bmm_5: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59)
        view_60: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [64, 4, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_11: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_11, [64, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_61, [32768, 256]);  view_61 = None
        permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        addmm_16: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_51, view_62, permute_31);  primals_51 = permute_31 = None
        view_63: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_16, [64, 512, 256]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_28: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_38: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_8, view_63);  view_63 = None
        mul_39: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_39, add_21);  mul_39 = add_21 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_10: "f32[64, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[64, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_24: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_8: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
        mul_40: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_41: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_40, primals_52)
        add_25: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_41, primals_53);  mul_41 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_25, [32768, 256])
        permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_54, [1, 0])
        addmm_17: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_55, view_64, permute_32);  primals_55 = permute_32 = None
        view_65: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_17, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_42: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_43: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_44: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_42, add_26);  mul_42 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_66: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_44, [32768, 1024]);  mul_44 = None
        permute_33: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        addmm_18: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_57, view_66, permute_33);  primals_57 = permute_33 = None
        view_67: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_18, [64, 512, 256]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_27: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_45: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_9, view_67);  view_67 = None
        mul_46: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_46, add_25);  mul_46 = add_25 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_12: "f32[64, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[64, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_9: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_27, getitem_13);  add_27 = getitem_13 = None
        mul_47: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_48: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_47, primals_58)
        add_29: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_48, primals_59);  mul_48 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_68: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_29, [32768, 256])
        permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(primals_60, [1, 0])
        addmm_19: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_61, view_68, permute_34);  primals_61 = permute_34 = None
        view_69: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_19, [64, 512, 256]);  addmm_19 = None
        view_70: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_69, [64, 512, -1, 64]);  view_69 = None
        permute_35: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm_20: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_63, view_68, permute_36);  primals_63 = permute_36 = None
        view_72: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_20, [64, 512, 256]);  addmm_20 = None
        view_73: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_72, [64, 512, -1, 64]);  view_72 = None
        permute_37: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_38: "f32[256, 256]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_21: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_65, view_68, permute_38);  primals_65 = permute_38 = None
        view_75: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_21, [64, 512, 256]);  addmm_21 = None
        view_76: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_75, [64, 512, -1, 64]);  view_75 = None
        permute_39: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_49: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_35, 0.3535533905932738);  permute_35 = None
        permute_40: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        mul_50: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_40, 0.3535533905932738);  permute_40 = None
        expand_15: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_49, [64, 4, 512, 64]);  mul_49 = None
        clone_12: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_77: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_12, [256, 512, 64]);  clone_12 = None
        expand_16: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_50, [64, 4, 64, 512]);  mul_50 = None
        clone_13: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_78: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_13, [256, 64, 512]);  clone_13 = None
        bmm_6: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
        view_79: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [64, 4, 512, 512]);  bmm_6 = None
        add_30: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_79, where);  view_79 = None
        amax_3: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  amax_3 = None
        exp_3: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        eq_3: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_30, -inf);  add_30 = None
        logical_not_6: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_2, div_3);  logical_not_7 = div_3 = None
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_26: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_51: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_10, where_7)
        mul_52: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        expand_17: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_52, [64, 4, 512, 512]);  mul_52 = None
        view_80: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [256, 512, 512]);  expand_17 = None
        expand_18: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_39, [64, 4, 512, 64]);  permute_39 = None
        clone_14: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_81: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_14, [256, 512, 64]);  clone_14 = None
        bmm_7: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81)
        view_82: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [64, 4, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_15: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_83: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_15, [64, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_83, [32768, 256]);  view_83 = None
        permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        addmm_22: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_67, view_84, permute_42);  primals_67 = permute_42 = None
        view_85: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_22, [64, 512, 256]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_25: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_53: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_11, view_85);  view_85 = None
        mul_54: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_54, add_29);  mul_54 = add_29 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_14: "f32[64, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[64, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_32: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_11: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_31, getitem_15);  add_31 = getitem_15 = None
        mul_55: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_56: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_55, primals_68)
        add_33: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_56, primals_69);  mul_56 = primals_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_33, [32768, 256])
        permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        addmm_23: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_71, view_86, permute_43);  primals_71 = permute_43 = None
        view_87: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_23, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_57: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_58: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_59: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_57, add_34);  mul_57 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_88: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_59, [32768, 1024]);  mul_59 = None
        permute_44: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        addmm_24: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_73, view_88, permute_44);  primals_73 = permute_44 = None
        view_89: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_24, [64, 512, 256]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_24: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_60: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_12, view_89);  view_89 = None
        mul_61: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_61, add_33);  mul_61 = add_33 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_16: "f32[64, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[64, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_12: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_35, getitem_17);  add_35 = getitem_17 = None
        mul_62: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_63: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_62, primals_74)
        add_37: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_63, primals_75);  mul_63 = primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_90: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_37, [32768, 256])
        permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_25: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_77, view_90, permute_45);  primals_77 = permute_45 = None
        view_91: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_25, [64, 512, 256]);  addmm_25 = None
        view_92: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_91, [64, 512, -1, 64]);  view_91 = None
        permute_46: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        addmm_26: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_79, view_90, permute_47);  primals_79 = permute_47 = None
        view_94: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_26, [64, 512, 256]);  addmm_26 = None
        view_95: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_94, [64, 512, -1, 64]);  view_94 = None
        permute_48: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_49: "f32[256, 256]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_27: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_81, view_90, permute_49);  primals_81 = permute_49 = None
        view_97: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_27, [64, 512, 256]);  addmm_27 = None
        view_98: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_97, [64, 512, -1, 64]);  view_97 = None
        permute_50: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_64: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_46, 0.3535533905932738);  permute_46 = None
        permute_51: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        mul_65: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_51, 0.3535533905932738);  permute_51 = None
        expand_19: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_64, [64, 4, 512, 64]);  mul_64 = None
        clone_16: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_99: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [256, 512, 64]);  clone_16 = None
        expand_20: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_65, [64, 4, 64, 512]);  mul_65 = None
        clone_17: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_100: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_17, [256, 64, 512]);  clone_17 = None
        bmm_8: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100)
        view_101: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [64, 4, 512, 512]);  bmm_8 = None
        add_38: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_101, where);  view_101 = None
        amax_4: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
        sub_13: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  amax_4 = None
        exp_4: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        eq_4: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_38, -inf);  add_38 = None
        logical_not_8: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_2, div_4);  logical_not_9 = div_4 = None
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_23: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_66: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_13, where_9)
        mul_67: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        expand_21: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_67, [64, 4, 512, 512]);  mul_67 = None
        view_102: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [256, 512, 512]);  expand_21 = None
        expand_22: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_50, [64, 4, 512, 64]);  permute_50 = None
        clone_18: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_103: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [256, 512, 64]);  clone_18 = None
        bmm_9: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103)
        view_104: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [64, 4, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_19: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_19, [64, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_105, [32768, 256]);  view_105 = None
        permute_53: "f32[256, 256]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        addmm_28: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_83, view_106, permute_53);  primals_83 = permute_53 = None
        view_107: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_28, [64, 512, 256]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_22: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_68: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_14, view_107);  view_107 = None
        mul_69: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_69, add_37);  mul_69 = add_37 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_18: "f32[64, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[64, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_40: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_14: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_39, getitem_19);  add_39 = getitem_19 = None
        mul_70: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_71: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_70, primals_84)
        add_41: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_71, primals_85);  mul_71 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_41, [32768, 256])
        permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        addmm_29: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_87, view_108, permute_54);  primals_87 = permute_54 = None
        view_109: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_29, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_72: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_73: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_74: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_72, add_42);  mul_72 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_74, [32768, 1024]);  mul_74 = None
        permute_55: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        addmm_30: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_89, view_110, permute_55);  primals_89 = permute_55 = None
        view_111: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_30, [64, 512, 256]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_21: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_75: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_15, view_111);  view_111 = None
        mul_76: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_76, add_41);  mul_76 = add_41 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_20: "f32[64, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[64, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_15: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_43, getitem_21);  add_43 = getitem_21 = None
        mul_77: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_78: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_77, primals_90)
        add_45: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_78, primals_91);  mul_78 = primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_112: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_45, [32768, 256])
        permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        addmm_31: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_93, view_112, permute_56);  primals_93 = permute_56 = None
        view_113: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_31, [64, 512, 256]);  addmm_31 = None
        view_114: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_113, [64, 512, -1, 64]);  view_113 = None
        permute_57: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_58: "f32[256, 256]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_32: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_95, view_112, permute_58);  primals_95 = permute_58 = None
        view_116: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_32, [64, 512, 256]);  addmm_32 = None
        view_117: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_116, [64, 512, -1, 64]);  view_116 = None
        permute_59: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_60: "f32[256, 256]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_33: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_97, view_112, permute_60);  primals_97 = permute_60 = None
        view_119: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_33, [64, 512, 256]);  addmm_33 = None
        view_120: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_119, [64, 512, -1, 64]);  view_119 = None
        permute_61: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_79: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_57, 0.3535533905932738);  permute_57 = None
        permute_62: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        mul_80: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_62, 0.3535533905932738);  permute_62 = None
        expand_23: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_79, [64, 4, 512, 64]);  mul_79 = None
        clone_20: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_121: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_20, [256, 512, 64]);  clone_20 = None
        expand_24: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_80, [64, 4, 64, 512]);  mul_80 = None
        clone_21: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_122: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_21, [256, 64, 512]);  clone_21 = None
        bmm_10: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122)
        view_123: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [64, 4, 512, 512]);  bmm_10 = None
        add_46: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_123, where);  view_123 = None
        amax_5: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
        sub_16: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  amax_5 = None
        exp_5: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        eq_5: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_46, -inf);  add_46 = None
        logical_not_10: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_2, div_5);  logical_not_11 = div_5 = None
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_20: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_81: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_16, where_11)
        mul_82: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_81, 1.1111111111111112);  mul_81 = None
        expand_25: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_82, [64, 4, 512, 512]);  mul_82 = None
        view_124: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [256, 512, 512]);  expand_25 = None
        expand_26: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_61, [64, 4, 512, 64]);  permute_61 = None
        clone_22: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_125: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_22, [256, 512, 64]);  clone_22 = None
        bmm_11: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125)
        view_126: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [64, 4, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_23: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_127: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_23, [64, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_127, [32768, 256]);  view_127 = None
        permute_64: "f32[256, 256]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        addmm_34: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_99, view_128, permute_64);  primals_99 = permute_64 = None
        view_129: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_34, [64, 512, 256]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_19: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_83: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_17, view_129);  view_129 = None
        mul_84: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_84, add_45);  mul_84 = add_45 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_22: "f32[64, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[64, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_48: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_17: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_47, getitem_23);  add_47 = getitem_23 = None
        mul_85: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_86: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, primals_100)
        add_49: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_86, primals_101);  mul_86 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_49, [32768, 256])
        permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_102, [1, 0])
        addmm_35: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_103, view_130, permute_65);  primals_103 = permute_65 = None
        view_131: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_35, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_87: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_88: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_89: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, add_50);  mul_87 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_132: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_89, [32768, 1024]);  mul_89 = None
        permute_66: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        addmm_36: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_105, view_132, permute_66);  primals_105 = permute_66 = None
        view_133: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_36, [64, 512, 256]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_18: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_90: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_18, view_133);  view_133 = None
        mul_91: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_91, add_49);  mul_91 = add_49 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_24: "f32[64, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[64, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_18: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_51, getitem_25);  add_51 = getitem_25 = None
        mul_92: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_93: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_92, primals_106)
        add_53: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_93, primals_107);  mul_93 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_134: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_53, [32768, 256])
        permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(primals_108, [1, 0])
        addmm_37: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_109, view_134, permute_67);  primals_109 = permute_67 = None
        view_135: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_37, [64, 512, 256]);  addmm_37 = None
        view_136: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_135, [64, 512, -1, 64]);  view_135 = None
        permute_68: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_69: "f32[256, 256]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_38: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_111, view_134, permute_69);  primals_111 = permute_69 = None
        view_138: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_38, [64, 512, 256]);  addmm_38 = None
        view_139: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_138, [64, 512, -1, 64]);  view_138 = None
        permute_70: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_71: "f32[256, 256]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_39: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_113, view_134, permute_71);  primals_113 = permute_71 = None
        view_141: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_39, [64, 512, 256]);  addmm_39 = None
        view_142: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_141, [64, 512, -1, 64]);  view_141 = None
        permute_72: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_94: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_68, 0.3535533905932738);  permute_68 = None
        permute_73: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        mul_95: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_73, 0.3535533905932738);  permute_73 = None
        expand_27: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_94, [64, 4, 512, 64]);  mul_94 = None
        clone_24: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_143: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_24, [256, 512, 64]);  clone_24 = None
        expand_28: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_95, [64, 4, 64, 512]);  mul_95 = None
        clone_25: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_144: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_25, [256, 64, 512]);  clone_25 = None
        bmm_12: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144)
        view_145: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [64, 4, 512, 512]);  bmm_12 = None
        add_54: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_145, where);  view_145 = None
        amax_6: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
        sub_19: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  amax_6 = None
        exp_6: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        eq_6: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_54, -inf);  add_54 = None
        logical_not_12: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_13: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_2, div_6);  logical_not_13 = div_6 = None
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_17: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_96: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_19, where_13)
        mul_97: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        expand_29: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_97, [64, 4, 512, 512]);  mul_97 = None
        view_146: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [256, 512, 512]);  expand_29 = None
        expand_30: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_72, [64, 4, 512, 64]);  permute_72 = None
        clone_26: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_147: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [256, 512, 64]);  clone_26 = None
        bmm_13: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147)
        view_148: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [64, 4, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_27: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_149: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_27, [64, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_149, [32768, 256]);  view_149 = None
        permute_75: "f32[256, 256]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        addmm_40: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_115, view_150, permute_75);  primals_115 = permute_75 = None
        view_151: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_40, [64, 512, 256]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_16: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_98: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_20, view_151);  view_151 = None
        mul_99: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_99, add_53);  mul_99 = add_53 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_26: "f32[64, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[64, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_56: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_20: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_55, getitem_27);  add_55 = getitem_27 = None
        mul_100: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_101: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_100, primals_116)
        add_57: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_101, primals_117);  mul_101 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_57, [32768, 256])
        permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        addmm_41: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_119, view_152, permute_76);  primals_119 = permute_76 = None
        view_153: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_41, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_102: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_103: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_104: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_102, add_58);  mul_102 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_104, [32768, 1024]);  mul_104 = None
        permute_77: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        addmm_42: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_121, view_154, permute_77);  primals_121 = permute_77 = None
        view_155: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_42, [64, 512, 256]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_15: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_105: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_21, view_155);  view_155 = None
        mul_106: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_106, add_57);  mul_106 = add_57 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_28: "f32[64, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[64, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_21: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_59, getitem_29);  add_59 = getitem_29 = None
        mul_107: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_108: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_107, primals_122)
        add_61: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_108, primals_123);  mul_108 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_156: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_61, [32768, 256])
        permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        addmm_43: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_125, view_156, permute_78);  primals_125 = permute_78 = None
        view_157: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_43, [64, 512, 256]);  addmm_43 = None
        view_158: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_157, [64, 512, -1, 64]);  view_157 = None
        permute_79: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_80: "f32[256, 256]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_44: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_127, view_156, permute_80);  primals_127 = permute_80 = None
        view_160: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_44, [64, 512, 256]);  addmm_44 = None
        view_161: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_160, [64, 512, -1, 64]);  view_160 = None
        permute_81: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_82: "f32[256, 256]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_45: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_129, view_156, permute_82);  primals_129 = permute_82 = None
        view_163: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_45, [64, 512, 256]);  addmm_45 = None
        view_164: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_163, [64, 512, -1, 64]);  view_163 = None
        permute_83: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_109: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_79, 0.3535533905932738);  permute_79 = None
        permute_84: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        mul_110: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
        expand_31: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_109, [64, 4, 512, 64]);  mul_109 = None
        clone_28: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_165: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_28, [256, 512, 64]);  clone_28 = None
        expand_32: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_110, [64, 4, 64, 512]);  mul_110 = None
        clone_29: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_166: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_29, [256, 64, 512]);  clone_29 = None
        bmm_14: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166)
        view_167: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [64, 4, 512, 512]);  bmm_14 = None
        add_62: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_167, where);  view_167 = None
        amax_7: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
        sub_22: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  amax_7 = None
        exp_7: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        eq_7: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_62, -inf);  add_62 = None
        logical_not_14: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_15: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_2, div_7);  logical_not_15 = div_7 = None
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_14: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_111: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_22, where_15)
        mul_112: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        expand_33: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_112, [64, 4, 512, 512]);  mul_112 = None
        view_168: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [256, 512, 512]);  expand_33 = None
        expand_34: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_83, [64, 4, 512, 64]);  permute_83 = None
        clone_30: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_169: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_30, [256, 512, 64]);  clone_30 = None
        bmm_15: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169)
        view_170: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [64, 4, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_31: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_171: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_31, [64, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_171, [32768, 256]);  view_171 = None
        permute_86: "f32[256, 256]" = torch.ops.aten.permute.default(primals_130, [1, 0])
        addmm_46: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_131, view_172, permute_86);  primals_131 = permute_86 = None
        view_173: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_46, [64, 512, 256]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_13: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_113: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_23, view_173);  view_173 = None
        mul_114: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_114, add_61);  mul_114 = add_61 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_30: "f32[64, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[64, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_64: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_23: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_63, getitem_31);  add_63 = getitem_31 = None
        mul_115: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_116: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_115, primals_132)
        add_65: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_116, primals_133);  mul_116 = primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_65, [32768, 256])
        permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        addmm_47: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_135, view_174, permute_87);  primals_135 = permute_87 = None
        view_175: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_47, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_117: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_118: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_119: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_117, add_66);  mul_117 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_176: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_119, [32768, 1024]);  mul_119 = None
        permute_88: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        addmm_48: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_137, view_176, permute_88);  primals_137 = permute_88 = None
        view_177: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_48, [64, 512, 256]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_12: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_120: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_24, view_177);  view_177 = None
        mul_121: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_121, add_65);  mul_121 = add_65 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_32: "f32[64, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[64, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_24: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_67, getitem_33);  add_67 = getitem_33 = None
        mul_122: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_123: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_122, primals_138)
        add_69: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_123, primals_139);  mul_123 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_178: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_69, [32768, 256])
        permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_49: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_141, view_178, permute_89);  primals_141 = permute_89 = None
        view_179: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_49, [64, 512, 256]);  addmm_49 = None
        view_180: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_179, [64, 512, -1, 64]);  view_179 = None
        permute_90: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_91: "f32[256, 256]" = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_50: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_143, view_178, permute_91);  primals_143 = permute_91 = None
        view_182: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_50, [64, 512, 256]);  addmm_50 = None
        view_183: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_182, [64, 512, -1, 64]);  view_182 = None
        permute_92: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_93: "f32[256, 256]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        addmm_51: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_145, view_178, permute_93);  primals_145 = permute_93 = None
        view_185: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_51, [64, 512, 256]);  addmm_51 = None
        view_186: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_185, [64, 512, -1, 64]);  view_185 = None
        permute_94: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_124: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_90, 0.3535533905932738);  permute_90 = None
        permute_95: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        mul_125: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_95, 0.3535533905932738);  permute_95 = None
        expand_35: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_124, [64, 4, 512, 64]);  mul_124 = None
        clone_32: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_187: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_32, [256, 512, 64]);  clone_32 = None
        expand_36: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_125, [64, 4, 64, 512]);  mul_125 = None
        clone_33: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_188: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_33, [256, 64, 512]);  clone_33 = None
        bmm_16: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188)
        view_189: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [64, 4, 512, 512]);  bmm_16 = None
        add_70: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_189, where);  view_189 = None
        amax_8: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
        sub_25: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  amax_8 = None
        exp_8: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        eq_8: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_70, -inf);  add_70 = None
        logical_not_16: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_17: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_2, div_8);  logical_not_17 = div_8 = None
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_11: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_126: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_25, where_17)
        mul_127: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_126, 1.1111111111111112);  mul_126 = None
        expand_37: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_127, [64, 4, 512, 512]);  mul_127 = None
        view_190: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [256, 512, 512]);  expand_37 = None
        expand_38: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_94, [64, 4, 512, 64]);  permute_94 = None
        clone_34: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_191: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_34, [256, 512, 64]);  clone_34 = None
        bmm_17: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
        view_192: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [64, 4, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_35: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_35, [64, 512, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_193, [32768, 256]);  view_193 = None
        permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(primals_146, [1, 0])
        addmm_52: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_147, view_194, permute_97);  primals_147 = permute_97 = None
        view_195: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_52, [64, 512, 256]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_10: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_128: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_26, view_195);  view_195 = None
        mul_129: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_129, add_69);  mul_129 = add_69 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_34: "f32[64, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[64, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_72: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_26: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_71, getitem_35);  add_71 = getitem_35 = None
        mul_130: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_131: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_130, primals_148)
        add_73: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_131, primals_149);  mul_131 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_73, [32768, 256])
        permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        addmm_53: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_151, view_196, permute_98);  primals_151 = permute_98 = None
        view_197: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_53, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_132: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_133: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_8: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_134: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_132, add_74);  mul_132 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_198: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_134, [32768, 1024]);  mul_134 = None
        permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_152, [1, 0])
        addmm_54: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_153, view_198, permute_99);  primals_153 = permute_99 = None
        view_199: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_54, [64, 512, 256]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_9: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_27: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_135: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_27, view_199);  view_199 = None
        mul_136: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_136, add_73);  mul_136 = add_73 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_36: "f32[64, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[64, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_27: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_75, getitem_37);  add_75 = getitem_37 = None
        mul_137: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_138: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, primals_154)
        add_77: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_138, primals_155);  mul_138 = primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_200: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_77, [32768, 256])
        permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_55: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_157, view_200, permute_100);  primals_157 = permute_100 = None
        view_201: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_55, [64, 512, 256]);  addmm_55 = None
        view_202: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_201, [64, 512, -1, 64]);  view_201 = None
        permute_101: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_102: "f32[256, 256]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_56: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_159, view_200, permute_102);  primals_159 = permute_102 = None
        view_204: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_56, [64, 512, 256]);  addmm_56 = None
        view_205: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_204, [64, 512, -1, 64]);  view_204 = None
        permute_103: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_104: "f32[256, 256]" = torch.ops.aten.permute.default(primals_160, [1, 0])
        addmm_57: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_161, view_200, permute_104);  primals_161 = permute_104 = None
        view_207: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_57, [64, 512, 256]);  addmm_57 = None
        view_208: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_207, [64, 512, -1, 64]);  view_207 = None
        permute_105: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_139: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_101, 0.3535533905932738);  permute_101 = None
        permute_106: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        mul_140: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_106, 0.3535533905932738);  permute_106 = None
        expand_39: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_139, [64, 4, 512, 64]);  mul_139 = None
        clone_36: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_209: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_36, [256, 512, 64]);  clone_36 = None
        expand_40: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_140, [64, 4, 64, 512]);  mul_140 = None
        clone_37: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_210: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_37, [256, 64, 512]);  clone_37 = None
        bmm_18: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210)
        view_211: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [64, 4, 512, 512]);  bmm_18 = None
        add_78: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_211, where);  view_211 = None
        amax_9: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_28: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  amax_9 = None
        exp_9: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        eq_9: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf);  add_78 = None
        logical_not_18: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_19: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_2, div_9);  logical_not_19 = div_9 = None
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_8: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_28: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_141: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_28, where_19)
        mul_142: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        expand_41: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_142, [64, 4, 512, 512]);  mul_142 = None
        view_212: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [256, 512, 512]);  expand_41 = None
        expand_42: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_105, [64, 4, 512, 64]);  permute_105 = None
        clone_38: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_213: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_38, [256, 512, 64]);  clone_38 = None
        bmm_19: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213)
        view_214: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [64, 4, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_39: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_39, [64, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_215, [32768, 256]);  view_215 = None
        permute_108: "f32[256, 256]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        addmm_58: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_163, view_216, permute_108);  primals_163 = permute_108 = None
        view_217: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_58, [64, 512, 256]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_7: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_29: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_143: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_29, view_217);  view_217 = None
        mul_144: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_144, add_77);  mul_144 = add_77 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_38: "f32[64, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[64, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_80: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_29: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_79, getitem_39);  add_79 = getitem_39 = None
        mul_145: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_146: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_145, primals_164)
        add_81: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_146, primals_165);  mul_146 = primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_81, [32768, 256])
        permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_166, [1, 0])
        addmm_59: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_167, view_218, permute_109);  primals_167 = permute_109 = None
        view_219: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_59, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_147: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_148: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_9: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_149: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_147, add_82);  mul_147 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_220: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_149, [32768, 1024]);  mul_149 = None
        permute_110: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_168, [1, 0])
        addmm_60: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_169, view_220, permute_110);  primals_169 = permute_110 = None
        view_221: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_60, [64, 512, 256]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_6: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_150: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_30, view_221);  view_221 = None
        mul_151: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_151, add_81);  mul_151 = add_81 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_40: "f32[64, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[64, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_30: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_83, getitem_41);  add_83 = getitem_41 = None
        mul_152: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_153: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_152, primals_170)
        add_85: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_153, primals_171);  mul_153 = primals_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_222: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_85, [32768, 256])
        permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_61: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_173, view_222, permute_111);  primals_173 = permute_111 = None
        view_223: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_61, [64, 512, 256]);  addmm_61 = None
        view_224: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_223, [64, 512, -1, 64]);  view_223 = None
        permute_112: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_113: "f32[256, 256]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_62: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_175, view_222, permute_113);  primals_175 = permute_113 = None
        view_226: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_62, [64, 512, 256]);  addmm_62 = None
        view_227: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_226, [64, 512, -1, 64]);  view_226 = None
        permute_114: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_115: "f32[256, 256]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        addmm_63: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_177, view_222, permute_115);  primals_177 = permute_115 = None
        view_229: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_63, [64, 512, 256]);  addmm_63 = None
        view_230: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_229, [64, 512, -1, 64]);  view_229 = None
        permute_116: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_154: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_112, 0.3535533905932738);  permute_112 = None
        permute_117: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        mul_155: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_117, 0.3535533905932738);  permute_117 = None
        expand_43: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_154, [64, 4, 512, 64]);  mul_154 = None
        clone_40: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_231: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_40, [256, 512, 64]);  clone_40 = None
        expand_44: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_155, [64, 4, 64, 512]);  mul_155 = None
        clone_41: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_232: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_41, [256, 64, 512]);  clone_41 = None
        bmm_20: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232)
        view_233: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [64, 4, 512, 512]);  bmm_20 = None
        add_86: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_233, where);  view_233 = None
        amax_10: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_31: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  amax_10 = None
        exp_10: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        eq_10: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_86, -inf);  add_86 = None
        logical_not_20: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_21: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_2, div_10);  logical_not_21 = div_10 = None
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_5: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_31: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_156: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_31, where_21)
        mul_157: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None
        expand_45: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_157, [64, 4, 512, 512]);  mul_157 = None
        view_234: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [256, 512, 512]);  expand_45 = None
        expand_46: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_116, [64, 4, 512, 64]);  permute_116 = None
        clone_42: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_235: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_42, [256, 512, 64]);  clone_42 = None
        bmm_21: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235)
        view_236: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [64, 4, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_43: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_43, [64, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_237, [32768, 256]);  view_237 = None
        permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_64: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_179, view_238, permute_119);  primals_179 = permute_119 = None
        view_239: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_64, [64, 512, 256]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_4: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_32: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_158: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_32, view_239);  view_239 = None
        mul_159: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_158, 1.1111111111111112);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_159, add_85);  mul_159 = add_85 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_42: "f32[64, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[64, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_88: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_32: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_87, getitem_43);  add_87 = getitem_43 = None
        mul_160: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_161: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_160, primals_180)
        add_89: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_161, primals_181);  mul_161 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_89, [32768, 256])
        permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_65: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_183, view_240, permute_120);  primals_183 = permute_120 = None
        view_241: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_65, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_162: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_163: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
        erf_10: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_164: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, add_90);  mul_162 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_242: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_164, [32768, 1024]);  mul_164 = None
        permute_121: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_66: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_185, view_242, permute_121);  primals_185 = permute_121 = None
        view_243: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_66, [64, 512, 256]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_3: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_33: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_165: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_33, view_243);  view_243 = None
        mul_166: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_166, add_89);  mul_166 = add_89 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_44: "f32[64, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[64, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_33: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_91, getitem_45);  add_91 = getitem_45 = None
        mul_167: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_168: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_167, primals_186)
        add_93: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_168, primals_187);  mul_168 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_244: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_93, [32768, 256])
        permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_67: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_189, view_244, permute_122);  primals_189 = permute_122 = None
        view_245: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_67, [64, 512, 256]);  addmm_67 = None
        view_246: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_245, [64, 512, -1, 64]);  view_245 = None
        permute_123: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_124: "f32[256, 256]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_68: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_191, view_244, permute_124);  primals_191 = permute_124 = None
        view_248: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_68, [64, 512, 256]);  addmm_68 = None
        view_249: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_248, [64, 512, -1, 64]);  view_248 = None
        permute_125: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_126: "f32[256, 256]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_69: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_193, view_244, permute_126);  primals_193 = permute_126 = None
        view_251: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_69, [64, 512, 256]);  addmm_69 = None
        view_252: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_251, [64, 512, -1, 64]);  view_251 = None
        permute_127: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_169: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_123, 0.3535533905932738);  permute_123 = None
        permute_128: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        mul_170: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_128, 0.3535533905932738);  permute_128 = None
        expand_47: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_169, [64, 4, 512, 64]);  mul_169 = None
        clone_44: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_253: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_44, [256, 512, 64]);  clone_44 = None
        expand_48: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_170, [64, 4, 64, 512]);  mul_170 = None
        clone_45: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_254: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_45, [256, 64, 512]);  clone_45 = None
        bmm_22: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254)
        view_255: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [64, 4, 512, 512]);  bmm_22 = None
        add_94: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_255, where);  view_255 = where = None
        amax_11: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_34: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  amax_11 = None
        exp_11: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        eq_11: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_94, -inf);  add_94 = None
        logical_not_22: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_23: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_2, div_11);  logical_not_23 = full_default_2 = div_11 = None
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_2: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_171: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_34, where_23)
        mul_172: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        expand_49: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_172, [64, 4, 512, 512]);  mul_172 = None
        view_256: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [256, 512, 512]);  expand_49 = None
        expand_50: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_127, [64, 4, 512, 64]);  permute_127 = None
        clone_46: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_257: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_46, [256, 512, 64]);  clone_46 = None
        bmm_23: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257)
        view_258: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [64, 4, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_47: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_259: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_47, [64, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_259, [32768, 256]);  view_259 = None
        permute_130: "f32[256, 256]" = torch.ops.aten.permute.default(primals_194, [1, 0])
        addmm_70: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_195, view_260, permute_130);  primals_195 = permute_130 = None
        view_261: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_70, [64, 512, 256]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_1: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_35: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_173: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_35, view_261);  view_261 = None
        mul_174: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_174, add_93);  mul_174 = add_93 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_46: "f32[64, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[64, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_96: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_35: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_95, getitem_47);  add_95 = getitem_47 = None
        mul_175: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_176: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_175, primals_196)
        add_97: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_176, primals_197);  mul_176 = primals_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_97, [32768, 256])
        permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_71: "f32[32768, 1024]" = torch.ops.aten.addmm.default(primals_199, view_262, permute_131);  primals_199 = permute_131 = None
        view_263: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_71, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_177: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_178: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_11: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_179: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_177, add_98);  mul_177 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_179, [32768, 1024]);  mul_179 = None
        permute_132: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_200, [1, 0])
        addmm_72: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_201, view_264, permute_132);  primals_201 = permute_132 = None
        view_265: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_72, [64, 512, 256]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 512, 256]" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_36: "b8[64, 512, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_180: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(gt_36, view_265);  view_265 = None
        mul_181: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_181, add_97);  mul_181 = add_97 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_48: "f32[64, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[64, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_36: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_99, getitem_49);  add_99 = getitem_49 = None
        mul_182: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_183: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_182, primals_202)
        add_101: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_183, primals_203);  mul_183 = primals_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_266: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_101, [32768, 256]);  add_101 = None
        permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(primals_204, [1, 0])
        addmm_73: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_205, view_266, permute_133);  primals_205 = permute_133 = None
        view_267: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [64, 512, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_184: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_185: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
        erf_12: "f32[64, 512, 128]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_186, [2], correction = 0, keepdim = True)
        getitem_50: "f32[64, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[64, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_103: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_37: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_186, getitem_51);  mul_186 = None
        mul_187: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_188: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_187, primals_206);  mul_187 = None
        add_104: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_188, primals_207);  mul_188 = primals_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        view_268: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_104, [32768, 128]);  add_104 = None
        permute_134: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        constant_pad_nd_default_2: "f32[128, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 2, 0, 0]);  permute_134 = None
        full_default_47: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([primals_208, full_default_47]);  primals_208 = full_default_47 = None
        addmm_default: "f32[32768, 30524]" = torch.ops.aten.addmm.default(cat_default, view_268, constant_pad_nd_default_2);  cat_default = constant_pad_nd_default_2 = None
        slice_tensor: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_269: "f32[64, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, [64, 512, 30522]);  slice_tensor = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[64, 513]" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1], -100.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[64, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone_48: "i64[64, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_270: "f32[32768, 30522]" = torch.ops.aten.reshape.default(view_269, [-1, 30522])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_271: "i64[32768]" = torch.ops.aten.reshape.default(clone_48, [-1]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_12: "f32[32768, 1]" = torch.ops.aten.amax.default(view_270, [1], True)
        sub_38: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = None
        exp_12: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_271, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[32768]" = torch.ops.aten.where.self(ne, view_271, full_default_36);  view_271 = full_default_36 = None
        unsqueeze_3: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_3);  sub_39 = unsqueeze_3 = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_25: "f32[32768]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_16: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_156: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_157: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
        permute_158: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_159: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_18: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_189: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        permute_190: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_191: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_192: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_20: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_222: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        permute_223: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_224: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_225: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_22: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_255: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        permute_256: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_257: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_258: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_24: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 256);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_288: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_289: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_290: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_291: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_25: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_26: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 256);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_321: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
        permute_322: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_323: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        permute_324: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_27: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_28: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_354: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_355: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_356: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        permute_357: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_29: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_30: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_387: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        permute_388: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_389: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_390: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_31: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_32: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_420: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        permute_421: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        permute_422: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_423: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_33: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_34: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_453: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_454: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_455: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_456: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_35: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_486: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        permute_487: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_488: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_489: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_37: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_38: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_519: "f32[256, 512, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_520: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_521: "f32[256, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_522: "f32[256, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_39: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (div_12, view_269, primals_2, primals_3, primals_5, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, gather, mul, gt, view, ge, view_2, bmm, amax, sum_1, logical_not_1, gt_1, view_18, gt_2, mul_10, view_20, addmm_5, view_22, gt_3, mul_17, view_24, where_3, gt_4, view_40, gt_5, mul_25, view_42, addmm_11, view_44, gt_6, mul_32, view_46, where_5, gt_7, view_62, gt_8, mul_40, view_64, addmm_17, view_66, gt_9, mul_47, view_68, where_7, gt_10, view_84, gt_11, mul_55, view_86, addmm_23, view_88, gt_12, mul_62, view_90, where_9, gt_13, view_106, gt_14, mul_70, view_108, addmm_29, view_110, gt_15, mul_77, view_112, where_11, gt_16, view_128, gt_17, mul_85, view_130, addmm_35, view_132, gt_18, mul_92, view_134, where_13, gt_19, view_150, gt_20, mul_100, view_152, addmm_41, view_154, gt_21, mul_107, view_156, where_15, gt_22, view_172, gt_23, mul_115, view_174, addmm_47, view_176, gt_24, mul_122, view_178, where_17, gt_25, view_194, gt_26, mul_130, view_196, addmm_53, view_198, gt_27, mul_137, view_200, where_19, gt_28, view_216, gt_29, mul_145, view_218, addmm_59, view_220, gt_30, mul_152, view_222, where_21, gt_31, view_238, gt_32, mul_160, view_240, addmm_65, view_242, gt_33, mul_167, view_244, where_23, gt_34, view_260, gt_35, mul_175, view_262, addmm_71, view_264, gt_36, mul_182, view_266, addmm_73, getitem_51, rsqrt_25, view_268, view_269, constant_pad_nd, amax_12, log, convert_element_type, div_15, div_16, permute_156, permute_157, permute_158, permute_159, div_17, div_18, permute_189, permute_190, permute_191, permute_192, div_19, div_20, permute_222, permute_223, permute_224, permute_225, div_21, div_22, permute_255, permute_256, permute_257, permute_258, div_23, div_24, permute_288, permute_289, permute_290, permute_291, div_25, div_26, permute_321, permute_322, permute_323, permute_324, div_27, div_28, permute_354, permute_355, permute_356, permute_357, div_29, div_30, permute_387, permute_388, permute_389, permute_390, div_31, div_32, permute_420, permute_421, permute_422, permute_423, div_33, div_34, permute_453, permute_454, permute_455, permute_456, div_35, div_36, permute_486, permute_487, permute_488, permute_489, div_37, div_38, permute_519, permute_520, permute_521, permute_522, div_39)
