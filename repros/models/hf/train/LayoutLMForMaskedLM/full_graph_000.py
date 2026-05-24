import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "f32[30522, 768]", primals_3: "i64[1, 512]", primals_4: "f32[512, 768]", primals_5: "f32[1024, 768]", primals_6: "f32[1024, 768]", primals_7: "f32[1024, 768]", primals_8: "f32[1024, 768]", primals_9: "f32[2, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768, 768]", primals_15: "f32[768]", primals_16: "f32[768, 768]", primals_17: "f32[768]", primals_18: "f32[768, 768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[3072, 768]", primals_23: "f32[3072]", primals_24: "f32[768, 3072]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768, 768]", primals_31: "f32[768]", primals_32: "f32[768, 768]", primals_33: "f32[768]", primals_34: "f32[768, 768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[3072, 768]", primals_39: "f32[3072]", primals_40: "f32[768, 3072]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768, 768]", primals_47: "f32[768]", primals_48: "f32[768, 768]", primals_49: "f32[768]", primals_50: "f32[768, 768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[3072, 768]", primals_55: "f32[3072]", primals_56: "f32[768, 3072]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768, 768]", primals_63: "f32[768]", primals_64: "f32[768, 768]", primals_65: "f32[768]", primals_66: "f32[768, 768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[3072, 768]", primals_71: "f32[3072]", primals_72: "f32[768, 3072]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768, 768]", primals_79: "f32[768]", primals_80: "f32[768, 768]", primals_81: "f32[768]", primals_82: "f32[768, 768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[3072, 768]", primals_87: "f32[3072]", primals_88: "f32[768, 3072]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768, 768]", primals_95: "f32[768]", primals_96: "f32[768, 768]", primals_97: "f32[768]", primals_98: "f32[768, 768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[3072, 768]", primals_103: "f32[3072]", primals_104: "f32[768, 3072]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[768, 768]", primals_109: "f32[768]", primals_110: "f32[768, 768]", primals_111: "f32[768]", primals_112: "f32[768, 768]", primals_113: "f32[768]", primals_114: "f32[768, 768]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[768]", primals_118: "f32[3072, 768]", primals_119: "f32[3072]", primals_120: "f32[768, 3072]", primals_121: "f32[768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768, 768]", primals_125: "f32[768]", primals_126: "f32[768, 768]", primals_127: "f32[768]", primals_128: "f32[768, 768]", primals_129: "f32[768]", primals_130: "f32[768, 768]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[3072, 768]", primals_135: "f32[3072]", primals_136: "f32[768, 3072]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768, 768]", primals_143: "f32[768]", primals_144: "f32[768, 768]", primals_145: "f32[768]", primals_146: "f32[768, 768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[3072, 768]", primals_151: "f32[3072]", primals_152: "f32[768, 3072]", primals_153: "f32[768]", primals_154: "f32[768]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[768, 768]", primals_159: "f32[768]", primals_160: "f32[768, 768]", primals_161: "f32[768]", primals_162: "f32[768, 768]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768]", primals_166: "f32[3072, 768]", primals_167: "f32[3072]", primals_168: "f32[768, 3072]", primals_169: "f32[768]", primals_170: "f32[768]", primals_171: "f32[768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768]", primals_176: "f32[768, 768]", primals_177: "f32[768]", primals_178: "f32[768, 768]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "f32[768]", primals_182: "f32[3072, 768]", primals_183: "f32[3072]", primals_184: "f32[768, 3072]", primals_185: "f32[768]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[768, 768]", primals_189: "f32[768]", primals_190: "f32[768, 768]", primals_191: "f32[768]", primals_192: "f32[768, 768]", primals_193: "f32[768]", primals_194: "f32[768, 768]", primals_195: "f32[768]", primals_196: "f32[768]", primals_197: "f32[768]", primals_198: "f32[3072, 768]", primals_199: "f32[3072]", primals_200: "f32[768, 3072]", primals_201: "f32[768]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768, 768]", primals_205: "f32[768]", primals_206: "f32[768, 768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[768]", primals_210: "f32[30522]", primals_211: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:498 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[32, 512]" = torch.ops.aten.full.default([32, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:501 in forward, code: bbox = torch.zeros(input_shape + (4,), dtype=torch.long, device=device)
        full_default_1: "i64[32, 512, 4]" = torch.ops.aten.full.default([32, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_3);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        select: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 0)
        embedding_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, select)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:98 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        select_1: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 1)
        embedding_3: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_6, select_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        select_2: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 2)
        embedding_4: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, select_2);  primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:100 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        select_3: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 3);  full_default_1 = None
        embedding_5: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_6, select_3);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        sub_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_3, select_1)
        embedding_6: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_7, sub_1);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        sub_2: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_2, select)
        embedding_7: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_8, sub_2);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_8: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_9, full_default);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_10)
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_11);  mul_2 = primals_11 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[25]" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:120 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_24: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_9);  add_9 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_4, [16384, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_13, view, permute);  primals_13 = permute = None
        view_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None
        view_2: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None
        permute_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_15, view, permute_2);  primals_15 = permute_2 = None
        view_4: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None
        view_5: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [32, 512, -1, 64]);  view_4 = None
        permute_3: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_17, view, permute_4);  primals_17 = permute_4 = None
        view_7: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 768]);  addmm_2 = None
        view_8: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [32, 512, -1, 64]);  view_7 = None
        permute_5: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # No stacktrace found for following nodes
        clone_default_33: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        clone_default_34: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        clone_default_35: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_129: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0]

        # No stacktrace found for following nodes
        getitem_130: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_11[1]
        getitem_131: "i64[]" = _scaled_dot_product_efficient_attention_default_11[2]
        getitem_132: "i64[]" = _scaled_dot_product_efficient_attention_default_11[3];  _scaled_dot_product_efficient_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3])
        clone_3: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [32, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_15, [16384, 768]);  view_15 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_19, view_16, permute_8);  primals_19 = permute_8 = None
        view_17: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_23: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_2: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, mul_4);  mul_9 = mul_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_12: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_3);  add_11 = getitem_3 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_20)
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_21);  mul_11 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_13, [16384, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        addmm_4: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_23, view_18, permute_9);  primals_23 = permute_9 = None
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_14: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_14);  mul_12 = add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_14, [16384, 3072]);  mul_14 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_25, view_20, permute_10);  primals_25 = permute_10 = None
        view_21: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_22: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_3: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, add_13);  mul_16 = add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_16: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_26)
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_27);  mul_18 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_22: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_17, [16384, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_29, view_22, permute_11);  primals_29 = permute_11 = None
        view_23: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None
        view_24: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [32, 512, -1, 64]);  view_23 = None
        permute_12: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_31, view_22, permute_13);  primals_31 = permute_13 = None
        view_26: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None
        view_27: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_26, [32, 512, -1, 64]);  view_26 = None
        permute_14: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_33, view_22, permute_15);  primals_33 = permute_15 = None
        view_29: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 768]);  addmm_8 = None
        view_30: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_29, [32, 512, -1, 64]);  view_29 = None
        permute_16: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # No stacktrace found for following nodes
        clone_default_30: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        clone_default_31: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        clone_default_32: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_122: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0]

        # No stacktrace found for following nodes
        getitem_123: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_10[1]
        getitem_124: "i64[]" = _scaled_dot_product_efficient_attention_default_10[2]
        getitem_125: "i64[]" = _scaled_dot_product_efficient_attention_default_10[3];  _scaled_dot_product_efficient_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3])
        clone_7: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [32, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_37, [16384, 768]);  view_37 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_35, view_38, permute_19);  primals_35 = permute_19 = None
        view_39: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_21: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_5: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, 1.1111111111111112);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, add_17);  mul_23 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_20: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_36)
        add_21: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_37);  mul_25 = primals_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_21, [16384, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_39, view_40, permute_20);  primals_39 = permute_20 = None
        view_41: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_26: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_22: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_22);  mul_26 = add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_28, [16384, 3072]);  mul_28 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_41, view_42, permute_21);  primals_41 = permute_21 = None
        view_43: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_20: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_6: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, 1.1111111111111112);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, add_21);  mul_30 = add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_24: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_9);  add_23 = getitem_9 = None
        mul_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = None
        mul_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, primals_42)
        add_25: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, primals_43);  mul_32 = primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_25, [16384, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_45, view_44, permute_22);  primals_45 = permute_22 = None
        view_45: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 768]);  addmm_12 = None
        view_46: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [32, 512, -1, 64]);  view_45 = None
        permute_23: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_47, view_44, permute_24);  primals_47 = permute_24 = None
        view_48: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None
        view_49: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_48, [32, 512, -1, 64]);  view_48 = None
        permute_25: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_49, view_44, permute_26);  primals_49 = permute_26 = None
        view_51: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 768]);  addmm_14 = None
        view_52: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_51, [32, 512, -1, 64]);  view_51 = None
        permute_27: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # No stacktrace found for following nodes
        clone_default_27: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        clone_default_28: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        clone_default_29: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_115: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0]

        # No stacktrace found for following nodes
        getitem_116: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_9[1]
        getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_default_9[2]
        getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_default_9[3];  _scaled_dot_product_efficient_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])
        clone_11: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [32, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_59, [16384, 768]);  view_59 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_51, view_60, permute_30);  primals_51 = permute_30 = None
        view_61: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_19: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_8: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_36: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_37: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, add_25);  mul_37 = add_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_28: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_11);  add_27 = getitem_11 = None
        mul_38: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = None
        mul_39: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_52)
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_53);  mul_39 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_29, [16384, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_54, [1, 0])
        addmm_16: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_55, view_62, permute_31);  primals_55 = permute_31 = None
        view_63: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_40: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_41: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_30: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_42: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_30);  mul_40 = add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_42, [16384, 3072]);  mul_42 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_57, view_64, permute_32);  primals_57 = permute_32 = None
        view_65: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_18: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_9: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_43: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_44: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, add_29);  mul_44 = add_29 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_32: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_13);  add_31 = getitem_13 = None
        mul_45: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = None
        mul_46: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_58)
        add_33: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_59);  mul_46 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_66: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_33, [16384, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0])
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_61, view_66, permute_33);  primals_61 = permute_33 = None
        view_67: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None
        view_68: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [32, 512, -1, 64]);  view_67 = None
        permute_34: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_63, view_66, permute_35);  primals_63 = permute_35 = None
        view_70: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None
        view_71: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_70, [32, 512, -1, 64]);  view_70 = None
        permute_36: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_65, view_66, permute_37);  primals_65 = permute_37 = None
        view_73: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None
        view_74: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None
        permute_38: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # No stacktrace found for following nodes
        clone_default_24: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        clone_default_25: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        clone_default_26: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_108: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0]

        # No stacktrace found for following nodes
        getitem_109: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_8[1]
        getitem_110: "i64[]" = _scaled_dot_product_efficient_attention_default_8[2]
        getitem_111: "i64[]" = _scaled_dot_product_efficient_attention_default_8[3];  _scaled_dot_product_efficient_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3])
        clone_15: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_15, [32, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_81, [16384, 768]);  view_81 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_67, view_82, permute_41);  primals_67 = permute_41 = None
        view_83: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_17: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_11: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_51: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, add_33);  mul_51 = add_33 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_15);  add_35 = getitem_15 = None
        mul_52: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = None
        mul_53: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_68)
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_69);  mul_53 = primals_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_37, [16384, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_71, view_84, permute_42);  primals_71 = permute_42 = None
        view_85: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_54: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_55: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_38: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_38);  mul_54 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_56, [16384, 3072]);  mul_56 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_73, view_86, permute_43);  primals_73 = permute_43 = None
        view_87: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_16: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_12: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_57: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_58: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, add_37);  mul_58 = add_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_40: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_17);  add_39 = getitem_17 = None
        mul_59: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_8);  sub_15 = None
        mul_60: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, primals_74)
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, primals_75);  mul_60 = primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_88: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_41, [16384, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_77, view_88, permute_44);  primals_77 = permute_44 = None
        view_89: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 768]);  addmm_24 = None
        view_90: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [32, 512, -1, 64]);  view_89 = None
        permute_45: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        addmm_25: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_79, view_88, permute_46);  primals_79 = permute_46 = None
        view_92: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None
        view_93: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_92, [32, 512, -1, 64]);  view_92 = None
        permute_47: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_81, view_88, permute_48);  primals_81 = permute_48 = None
        view_95: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768]);  addmm_26 = None
        view_96: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_95, [32, 512, -1, 64]);  view_95 = None
        permute_49: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # No stacktrace found for following nodes
        clone_default_21: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        clone_default_22: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        clone_default_23: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_101: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0]

        # No stacktrace found for following nodes
        getitem_102: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_7[1]
        getitem_103: "i64[]" = _scaled_dot_product_efficient_attention_default_7[2]
        getitem_104: "i64[]" = _scaled_dot_product_efficient_attention_default_7[3];  _scaled_dot_product_efficient_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        clone_19: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [32, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_103, [16384, 768]);  view_103 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        addmm_27: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_83, view_104, permute_52);  primals_83 = permute_52 = None
        view_105: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_15: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_14: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_65: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, add_41);  mul_65 = add_41 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_44: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_19);  add_43 = getitem_19 = None
        mul_66: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_9);  sub_17 = None
        mul_67: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_84)
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_85);  mul_67 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_45, [16384, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        addmm_28: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_87, view_106, permute_53);  primals_87 = permute_53 = None
        view_107: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_68: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_69: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_46: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_70: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_46);  mul_68 = add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_70, [16384, 3072]);  mul_70 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        addmm_29: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_89, view_108, permute_54);  primals_89 = permute_54 = None
        view_109: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_14: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_15: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_71: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_72: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, add_45);  mul_72 = add_45 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_48: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_21);  add_47 = getitem_21 = None
        mul_73: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_10);  sub_18 = None
        mul_74: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_90)
        add_49: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_91);  mul_74 = primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_49, [16384, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        addmm_30: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_93, view_110, permute_55);  primals_93 = permute_55 = None
        view_111: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 768]);  addmm_30 = None
        view_112: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None
        permute_56: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_31: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_95, view_110, permute_57);  primals_95 = permute_57 = None
        view_114: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 768]);  addmm_31 = None
        view_115: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_114, [32, 512, -1, 64]);  view_114 = None
        permute_58: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_32: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_97, view_110, permute_59);  primals_97 = permute_59 = None
        view_117: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None
        view_118: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_117, [32, 512, -1, 64]);  view_117 = None
        permute_60: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # No stacktrace found for following nodes
        clone_default_18: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        clone_default_19: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        clone_default_20: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_94: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0]

        # No stacktrace found for following nodes
        getitem_95: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_6[1]
        getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_default_6[2]
        getitem_97: "i64[]" = _scaled_dot_product_efficient_attention_default_6[3];  _scaled_dot_product_efficient_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3])
        clone_23: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_23, [32, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_125, [16384, 768]);  view_125 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        addmm_33: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_99, view_126, permute_63);  primals_99 = permute_63 = None
        view_127: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_13: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_17: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_78: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_79: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, add_49);  mul_79 = add_49 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_52: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_23);  add_51 = getitem_23 = None
        mul_80: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_11);  sub_20 = None
        mul_81: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_100)
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_101);  mul_81 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_53, [16384, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_102, [1, 0])
        addmm_34: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_103, view_128, permute_64);  primals_103 = permute_64 = None
        view_129: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_82: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_83: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_54: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_84: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_54);  mul_82 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_84, [16384, 3072]);  mul_84 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        addmm_35: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_105, view_130, permute_65);  primals_105 = permute_65 = None
        view_131: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_12: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_18: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_85: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_86: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, add_53);  mul_86 = add_53 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_56: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
        mul_87: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_12);  sub_21 = None
        mul_88: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_87, primals_106)
        add_57: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_88, primals_107);  mul_88 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_132: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_57, [16384, 768])
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0])
        addmm_36: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_109, view_132, permute_66);  primals_109 = permute_66 = None
        view_133: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 768]);  addmm_36 = None
        view_134: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [32, 512, -1, 64]);  view_133 = None
        permute_67: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_37: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_111, view_132, permute_68);  primals_111 = permute_68 = None
        view_136: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 768]);  addmm_37 = None
        view_137: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_136, [32, 512, -1, 64]);  view_136 = None
        permute_69: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_38: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_113, view_132, permute_70);  primals_113 = permute_70 = None
        view_139: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 768]);  addmm_38 = None
        view_140: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_139, [32, 512, -1, 64]);  view_139 = None
        permute_71: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # No stacktrace found for following nodes
        clone_default_15: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        clone_default_16: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        clone_default_17: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_87: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0]

        # No stacktrace found for following nodes
        getitem_88: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_5[1]
        getitem_89: "i64[]" = _scaled_dot_product_efficient_attention_default_5[2]
        getitem_90: "i64[]" = _scaled_dot_product_efficient_attention_default_5[3];  _scaled_dot_product_efficient_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_87, [0, 2, 1, 3])
        clone_27: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_27, [32, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_147, [16384, 768]);  view_147 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        addmm_39: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_115, view_148, permute_74);  primals_115 = permute_74 = None
        view_149: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_11: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_20: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_92: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, add_57);  mul_93 = add_57 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_60: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
        mul_94: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = None
        mul_95: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, primals_116)
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_95, primals_117);  mul_95 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_61, [16384, 768])
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        addmm_40: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_119, view_150, permute_75);  primals_119 = permute_75 = None
        view_151: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_96: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_97: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_62: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_98: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_62);  mul_96 = add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_98, [16384, 3072]);  mul_98 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        addmm_41: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_121, view_152, permute_76);  primals_121 = permute_76 = None
        view_153: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_10: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_21: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_99: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_100: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, add_61);  mul_100 = add_61 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_64: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_29);  add_63 = getitem_29 = None
        mul_101: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_14);  sub_24 = None
        mul_102: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_101, primals_122)
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_102, primals_123);  mul_102 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_154: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_65, [16384, 768])
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        addmm_42: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_125, view_154, permute_77);  primals_125 = permute_77 = None
        view_155: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 768]);  addmm_42 = None
        view_156: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [32, 512, -1, 64]);  view_155 = None
        permute_78: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_43: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_127, view_154, permute_79);  primals_127 = permute_79 = None
        view_158: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 768]);  addmm_43 = None
        view_159: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_158, [32, 512, -1, 64]);  view_158 = None
        permute_80: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_44: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_129, view_154, permute_81);  primals_129 = permute_81 = None
        view_161: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 768]);  addmm_44 = None
        view_162: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_161, [32, 512, -1, 64]);  view_161 = None
        permute_82: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # No stacktrace found for following nodes
        clone_default_12: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        clone_default_13: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        clone_default_14: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_80: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0]

        # No stacktrace found for following nodes
        getitem_81: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_4[1]
        getitem_82: "i64[]" = _scaled_dot_product_efficient_attention_default_4[2]
        getitem_83: "i64[]" = _scaled_dot_product_efficient_attention_default_4[3];  _scaled_dot_product_efficient_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3])
        clone_31: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_31, [32, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_169, [16384, 768]);  view_169 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0])
        addmm_45: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_131, view_170, permute_85);  primals_131 = permute_85 = None
        view_171: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_9: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_23: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_106: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_107: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_106, 1.1111111111111112);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_107, add_65);  mul_107 = add_65 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_68: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_31);  add_67 = getitem_31 = None
        mul_108: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = None
        mul_109: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, primals_132)
        add_69: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_109, primals_133);  mul_109 = primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_69, [16384, 768])
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        addmm_46: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_135, view_172, permute_86);  primals_135 = permute_86 = None
        view_173: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_110: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_111: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_70: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_112: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_110, add_70);  mul_110 = add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_112, [16384, 3072]);  mul_112 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        addmm_47: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_137, view_174, permute_87);  primals_137 = permute_87 = None
        view_175: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_8: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_24: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_113: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_114: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, add_69);  mul_114 = add_69 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_72: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_33);  add_71 = getitem_33 = None
        mul_115: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_16);  sub_27 = None
        mul_116: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, primals_138)
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_116, primals_139);  mul_116 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_176: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_73, [16384, 768])
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_48: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_141, view_176, permute_88);  primals_141 = permute_88 = None
        view_177: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None
        view_178: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [32, 512, -1, 64]);  view_177 = None
        permute_89: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_49: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_143, view_176, permute_90);  primals_143 = permute_90 = None
        view_180: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 768]);  addmm_49 = None
        view_181: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_180, [32, 512, -1, 64]);  view_180 = None
        permute_91: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        addmm_50: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_145, view_176, permute_92);  primals_145 = permute_92 = None
        view_183: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 768]);  addmm_50 = None
        view_184: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None
        permute_93: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # No stacktrace found for following nodes
        clone_default_9: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        clone_default_10: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        clone_default_11: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_73: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0]

        # No stacktrace found for following nodes
        getitem_74: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_3[1]
        getitem_75: "i64[]" = _scaled_dot_product_efficient_attention_default_3[2]
        getitem_76: "i64[]" = _scaled_dot_product_efficient_attention_default_3[3];  _scaled_dot_product_efficient_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_73, [0, 2, 1, 3])
        clone_35: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_35, [32, 512, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_191, [16384, 768]);  view_191 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0])
        addmm_51: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_147, view_192, permute_96);  primals_147 = permute_96 = None
        view_193: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_7: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_26: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_120: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_121: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_121, add_73);  mul_121 = add_73 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_76: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_29: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_35);  add_75 = getitem_35 = None
        mul_122: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_17);  sub_29 = None
        mul_123: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, primals_148)
        add_77: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_123, primals_149);  mul_123 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_77, [16384, 768])
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        addmm_52: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_151, view_194, permute_97);  primals_151 = permute_97 = None
        view_195: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_124: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_125: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_78: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_126: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_124, add_78);  mul_124 = add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_126, [16384, 3072]);  mul_126 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0])
        addmm_53: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_153, view_196, permute_98);  primals_153 = permute_98 = None
        view_197: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_6: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_27: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_127: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_128: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_128, add_77);  mul_128 = add_77 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_80: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_37);  add_79 = getitem_37 = None
        mul_129: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_18);  sub_30 = None
        mul_130: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_129, primals_154)
        add_81: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_130, primals_155);  mul_130 = primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_198: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_81, [16384, 768])
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_54: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_157, view_198, permute_99);  primals_157 = permute_99 = None
        view_199: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 768]);  addmm_54 = None
        view_200: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [32, 512, -1, 64]);  view_199 = None
        permute_100: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_55: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_159, view_198, permute_101);  primals_159 = permute_101 = None
        view_202: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None
        view_203: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_202, [32, 512, -1, 64]);  view_202 = None
        permute_102: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0])
        addmm_56: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_161, view_198, permute_103);  primals_161 = permute_103 = None
        view_205: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 768]);  addmm_56 = None
        view_206: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_205, [32, 512, -1, 64]);  view_205 = None
        permute_104: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # No stacktrace found for following nodes
        clone_default_6: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        clone_default_7: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        clone_default_8: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_66: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0]

        # No stacktrace found for following nodes
        getitem_67: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_2[1]
        getitem_68: "i64[]" = _scaled_dot_product_efficient_attention_default_2[2]
        getitem_69: "i64[]" = _scaled_dot_product_efficient_attention_default_2[3];  _scaled_dot_product_efficient_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3])
        clone_39: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_39, [32, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_213, [16384, 768]);  view_213 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        addmm_57: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_163, view_214, permute_107);  primals_163 = permute_107 = None
        view_215: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_5: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_29: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_134: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_135: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_135, add_81);  mul_135 = add_81 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_84: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_39);  add_83 = getitem_39 = None
        mul_136: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = None
        mul_137: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_136, primals_164)
        add_85: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_137, primals_165);  mul_137 = primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_85, [16384, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_166, [1, 0])
        addmm_58: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_167, view_216, permute_108);  primals_167 = permute_108 = None
        view_217: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_138: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_139: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_86: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_140: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_138, add_86);  mul_138 = add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_140, [16384, 3072]);  mul_140 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0])
        addmm_59: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_169, view_218, permute_109);  primals_169 = permute_109 = None
        view_219: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_4: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_30: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_141: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_142: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_142, add_85);  mul_142 = add_85 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_88: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_41);  add_87 = getitem_41 = None
        mul_143: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_20);  sub_33 = None
        mul_144: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_143, primals_170)
        add_89: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_144, primals_171);  mul_144 = primals_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_220: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_89, [16384, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_60: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_173, view_220, permute_110);  primals_173 = permute_110 = None
        view_221: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None
        view_222: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64]);  view_221 = None
        permute_111: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_61: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_175, view_220, permute_112);  primals_175 = permute_112 = None
        view_224: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 768]);  addmm_61 = None
        view_225: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_224, [32, 512, -1, 64]);  view_224 = None
        permute_113: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        addmm_62: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_177, view_220, permute_114);  primals_177 = permute_114 = None
        view_227: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None
        view_228: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_227, [32, 512, -1, 64]);  view_227 = None
        permute_115: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # No stacktrace found for following nodes
        clone_default_3: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        clone_default_4: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        clone_default_5: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_59: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0]

        # No stacktrace found for following nodes
        getitem_60: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default_1[1]
        getitem_61: "i64[]" = _scaled_dot_product_efficient_attention_default_1[2]
        getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_default_1[3];  _scaled_dot_product_efficient_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3])
        clone_43: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_43, [32, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_235, [16384, 768]);  view_235 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_63: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_179, view_236, permute_118);  primals_179 = permute_118 = None
        view_237: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_3: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_32: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_148: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_149: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_148, 1.1111111111111112);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_149, add_89);  mul_149 = add_89 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_92: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_43);  add_91 = getitem_43 = None
        mul_150: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_21);  sub_35 = None
        mul_151: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, primals_180)
        add_93: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_151, primals_181);  mul_151 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_93, [16384, 768])
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_64: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_183, view_238, permute_119);  primals_183 = permute_119 = None
        view_239: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_152: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_153: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_94: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_154: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_152, add_94);  mul_152 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_154, [16384, 3072]);  mul_154 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_65: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_185, view_240, permute_120);  primals_185 = permute_120 = None
        view_241: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_2: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_33: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_155: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_156: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_156, add_93);  mul_156 = add_93 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_96: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_45);  add_95 = getitem_45 = None
        mul_157: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = None
        mul_158: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_157, primals_186)
        add_97: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_158, primals_187);  mul_158 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_242: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_97, [16384, 768])
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_66: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_189, view_242, permute_121);  primals_189 = permute_121 = None
        view_243: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 768]);  addmm_66 = None
        view_244: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [32, 512, -1, 64]);  view_243 = None
        permute_122: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_67: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_191, view_242, permute_123);  primals_191 = permute_123 = None
        view_246: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        view_247: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_246, [32, 512, -1, 64]);  view_246 = None
        permute_124: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_68: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_193, view_242, permute_125);  primals_193 = permute_125 = None
        view_249: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 768]);  addmm_68 = None
        view_250: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_249, [32, 512, -1, 64]);  view_249 = None
        permute_126: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # No stacktrace found for following nodes
        clone_default: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        clone_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        clone_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_52: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default[0]

        # No stacktrace found for following nodes
        getitem_53: "f32[32, 12, 512]" = _scaled_dot_product_efficient_attention_default[1]
        getitem_54: "i64[]" = _scaled_dot_product_efficient_attention_default[2]
        getitem_55: "i64[]" = _scaled_dot_product_efficient_attention_default[3];  _scaled_dot_product_efficient_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3])
        clone_47: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_47, [32, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_257, [16384, 768]);  view_257 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0])
        addmm_69: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_195, view_258, permute_129);  primals_195 = permute_129 = None
        view_259: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_1: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_35: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_162: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_163: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_163, add_97);  mul_163 = add_97 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_100: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_38: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_47);  add_99 = getitem_47 = None
        mul_164: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_23);  sub_38 = None
        mul_165: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_164, primals_196)
        add_101: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_165, primals_197);  mul_165 = primals_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_101, [16384, 768])
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_70: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_199, view_260, permute_130);  primals_199 = permute_130 = None
        view_261: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_166: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_167: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_102: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_168: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_166, add_102);  mul_166 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_168, [16384, 3072]);  mul_168 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_200, [1, 0])
        addmm_71: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_201, view_262, permute_131);  primals_201 = permute_131 = None
        view_263: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_36: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_169: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_170: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_103: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_170, add_101);  mul_170 = add_101 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_104: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_39: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_49);  add_103 = getitem_49 = None
        mul_171: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = None
        mul_172: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_171, primals_202)
        add_105: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_172, primals_203);  mul_172 = primals_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_105, [16384, 768]);  add_105 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(primals_206, [1, 0])
        addmm_73: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_207, view_264, permute_133);  primals_207 = permute_133 = None
        view_265: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_73, [32, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_173: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
        mul_174: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476);  view_265 = None
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_106: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_175: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, add_106);  mul_173 = add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_175, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_107: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_40: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_175, getitem_51);  mul_175 = None
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = None
        mul_177: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, primals_208);  mul_176 = None
        add_108: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_177, primals_209);  mul_177 = primals_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        view_266: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_108, [16384, 768]);  add_108 = None
        permute_134: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        constant_pad_nd_default_3: "f32[768, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 2, 0, 0]);  permute_134 = None
        full_default_27: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([primals_210, full_default_27]);  primals_210 = full_default_27 = None
        addmm_default: "f32[16384, 30524]" = torch.ops.aten.addmm.default(cat_default, view_266, constant_pad_nd_default_3);  cat_default = constant_pad_nd_default_3 = None
        slice_tensor_1: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_267: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor_1, [32, 512, 30522]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        view_268: "f32[16384, 30522]" = torch.ops.aten.reshape.default(view_267, [-1, 30522])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:634 in forward, code: labels.view(-1),
        view_269: "i64[16384]" = torch.ops.aten.reshape.default(primals_211, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        amax_12: "f32[16384, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
        sub_41: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = None
        exp_12: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_41)
        sum_13: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_42: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_41, log);  sub_41 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384]" = torch.ops.aten.where.self(ne, view_269, full_default_3);  view_269 = full_default_3 = None
        unsqueeze_2: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_42, 1, unsqueeze_2);  sub_42 = unsqueeze_2 = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384]" = torch.ops.aten.where.self(ne, neg, full_default_4);  neg = full_default_4 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_16: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_18: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_20: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_22: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_24: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_25: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_26: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_27: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_28: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_29: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_30: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_31: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_32: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_33: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_34: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_35: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_37: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_38: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_39: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_12, view_267, primals_1, primals_2, primals_3, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_206, primals_208, primals_211, full_default, select, select_1, select_2, select_3, sub_1, sub_2, mul_1, gt, view, clone_default_33, clone_default_34, clone_default_35, getitem_129, getitem_130, getitem_131, getitem_132, view_16, gt_2, mul_10, view_18, addmm_4, view_20, gt_3, mul_17, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_122, getitem_123, getitem_124, getitem_125, view_38, gt_5, mul_24, view_40, addmm_10, view_42, gt_6, mul_31, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_115, getitem_116, getitem_117, getitem_118, view_60, gt_8, mul_38, view_62, addmm_16, view_64, gt_9, mul_45, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_108, getitem_109, getitem_110, getitem_111, view_82, gt_11, mul_52, view_84, addmm_22, view_86, gt_12, mul_59, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_101, getitem_102, getitem_103, getitem_104, view_104, gt_14, mul_66, view_106, addmm_28, view_108, gt_15, mul_73, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_94, getitem_95, getitem_96, getitem_97, view_126, gt_17, mul_80, view_128, addmm_34, view_130, gt_18, mul_87, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_87, getitem_88, getitem_89, getitem_90, view_148, gt_20, mul_94, view_150, addmm_40, view_152, gt_21, mul_101, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_80, getitem_81, getitem_82, getitem_83, view_170, gt_23, mul_108, view_172, addmm_46, view_174, gt_24, mul_115, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_73, getitem_74, getitem_75, getitem_76, view_192, gt_26, mul_122, view_194, addmm_52, view_196, gt_27, mul_129, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_66, getitem_67, getitem_68, getitem_69, view_214, gt_29, mul_136, view_216, addmm_58, view_218, gt_30, mul_143, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_59, getitem_60, getitem_61, getitem_62, view_236, gt_32, mul_150, view_238, addmm_64, view_240, gt_33, mul_157, view_242, clone_default, clone_default_1, clone_default_2, getitem_52, getitem_53, getitem_54, getitem_55, view_258, gt_35, mul_164, view_260, addmm_70, view_262, gt_36, mul_171, view_264, addmm_73, getitem_51, rsqrt_25, view_266, view_267, amax_12, log, convert_element_type, div_15, div_16, div_17, div_18, div_19, div_20, div_21, div_22, div_23, div_24, div_25, div_26, div_27, div_28, div_29, div_30, div_31, div_32, div_33, div_34, div_35, div_36, div_37, div_38, div_39)
