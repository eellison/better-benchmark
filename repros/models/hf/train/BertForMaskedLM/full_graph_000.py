class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512][512, 1]cuda:0", primals_2: "i64[1, 512][512, 1]cuda:0", primals_3: "i64[1, 512][512, 1]cuda:0", primals_4: "f32[30522, 768][768, 1]cuda:0", primals_5: "f32[2, 768][768, 1]cuda:0", primals_6: "f32[512, 768][768, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768][1]cuda:0", primals_9: "f32[768, 768][768, 1]cuda:0", primals_10: "f32[768][1]cuda:0", primals_11: "f32[768, 768][768, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768, 768][768, 1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[768, 768][768, 1]cuda:0", primals_16: "f32[768][1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_18: "f32[768][1]cuda:0", primals_19: "f32[3072, 768][768, 1]cuda:0", primals_20: "f32[3072][1]cuda:0", primals_21: "f32[768, 3072][3072, 1]cuda:0", primals_22: "f32[768][1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_24: "f32[768][1]cuda:0", primals_25: "f32[768, 768][768, 1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[768, 768][768, 1]cuda:0", primals_28: "f32[768][1]cuda:0", primals_29: "f32[768, 768][768, 1]cuda:0", primals_30: "f32[768][1]cuda:0", primals_31: "f32[768, 768][768, 1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_35: "f32[3072, 768][768, 1]cuda:0", primals_36: "f32[3072][1]cuda:0", primals_37: "f32[768, 3072][3072, 1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_39: "f32[768][1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_41: "f32[768, 768][768, 1]cuda:0", primals_42: "f32[768][1]cuda:0", primals_43: "f32[768, 768][768, 1]cuda:0", primals_44: "f32[768][1]cuda:0", primals_45: "f32[768, 768][768, 1]cuda:0", primals_46: "f32[768][1]cuda:0", primals_47: "f32[768, 768][768, 1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_50: "f32[768][1]cuda:0", primals_51: "f32[3072, 768][768, 1]cuda:0", primals_52: "f32[3072][1]cuda:0", primals_53: "f32[768, 3072][3072, 1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[768, 768][768, 1]cuda:0", primals_58: "f32[768][1]cuda:0", primals_59: "f32[768, 768][768, 1]cuda:0", primals_60: "f32[768][1]cuda:0", primals_61: "f32[768, 768][768, 1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_63: "f32[768, 768][768, 1]cuda:0", primals_64: "f32[768][1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_66: "f32[768][1]cuda:0", primals_67: "f32[3072, 768][768, 1]cuda:0", primals_68: "f32[3072][1]cuda:0", primals_69: "f32[768, 3072][3072, 1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_72: "f32[768][1]cuda:0", primals_73: "f32[768, 768][768, 1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_75: "f32[768, 768][768, 1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_77: "f32[768, 768][768, 1]cuda:0", primals_78: "f32[768][1]cuda:0", primals_79: "f32[768, 768][768, 1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_81: "f32[768][1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_83: "f32[3072, 768][768, 1]cuda:0", primals_84: "f32[3072][1]cuda:0", primals_85: "f32[768, 3072][3072, 1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_87: "f32[768][1]cuda:0", primals_88: "f32[768][1]cuda:0", primals_89: "f32[768, 768][768, 1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_91: "f32[768, 768][768, 1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_93: "f32[768, 768][768, 1]cuda:0", primals_94: "f32[768][1]cuda:0", primals_95: "f32[768, 768][768, 1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_100: "f32[3072][1]cuda:0", primals_101: "f32[768, 3072][3072, 1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_105: "f32[768, 768][768, 1]cuda:0", primals_106: "f32[768][1]cuda:0", primals_107: "f32[768, 768][768, 1]cuda:0", primals_108: "f32[768][1]cuda:0", primals_109: "f32[768, 768][768, 1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[768, 768][768, 1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_114: "f32[768][1]cuda:0", primals_115: "f32[3072, 768][768, 1]cuda:0", primals_116: "f32[3072][1]cuda:0", primals_117: "f32[768, 3072][3072, 1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_120: "f32[768][1]cuda:0", primals_121: "f32[768, 768][768, 1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_123: "f32[768, 768][768, 1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_125: "f32[768, 768][768, 1]cuda:0", primals_126: "f32[768][1]cuda:0", primals_127: "f32[768, 768][768, 1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_129: "f32[768][1]cuda:0", primals_130: "f32[768][1]cuda:0", primals_131: "f32[3072, 768][768, 1]cuda:0", primals_132: "f32[3072][1]cuda:0", primals_133: "f32[768, 3072][3072, 1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_135: "f32[768][1]cuda:0", primals_136: "f32[768][1]cuda:0", primals_137: "f32[768, 768][768, 1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768, 768][768, 1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[768, 768][768, 1]cuda:0", primals_142: "f32[768][1]cuda:0", primals_143: "f32[768, 768][768, 1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[3072, 768][768, 1]cuda:0", primals_148: "f32[3072][1]cuda:0", primals_149: "f32[768, 3072][3072, 1]cuda:0", primals_150: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768, 768][768, 1]cuda:0", primals_154: "f32[768][1]cuda:0", primals_155: "f32[768, 768][768, 1]cuda:0", primals_156: "f32[768][1]cuda:0", primals_157: "f32[768, 768][768, 1]cuda:0", primals_158: "f32[768][1]cuda:0", primals_159: "f32[768, 768][768, 1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_162: "f32[768][1]cuda:0", primals_163: "f32[3072, 768][768, 1]cuda:0", primals_164: "f32[3072][1]cuda:0", primals_165: "f32[768, 3072][3072, 1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_169: "f32[768, 768][768, 1]cuda:0", primals_170: "f32[768][1]cuda:0", primals_171: "f32[768, 768][768, 1]cuda:0", primals_172: "f32[768][1]cuda:0", primals_173: "f32[768, 768][768, 1]cuda:0", primals_174: "f32[768][1]cuda:0", primals_175: "f32[768, 768][768, 1]cuda:0", primals_176: "f32[768][1]cuda:0", primals_177: "f32[768][1]cuda:0", primals_178: "f32[768][1]cuda:0", primals_179: "f32[3072, 768][768, 1]cuda:0", primals_180: "f32[3072][1]cuda:0", primals_181: "f32[768, 3072][3072, 1]cuda:0", primals_182: "f32[768][1]cuda:0", primals_183: "f32[768][1]cuda:0", primals_184: "f32[768][1]cuda:0", primals_185: "f32[768, 768][768, 1]cuda:0", primals_186: "f32[768][1]cuda:0", primals_187: "f32[768, 768][768, 1]cuda:0", primals_188: "f32[768][1]cuda:0", primals_189: "f32[768, 768][768, 1]cuda:0", primals_190: "f32[768][1]cuda:0", primals_191: "f32[768, 768][768, 1]cuda:0", primals_192: "f32[768][1]cuda:0", primals_193: "f32[768][1]cuda:0", primals_194: "f32[768][1]cuda:0", primals_195: "f32[3072, 768][768, 1]cuda:0", primals_196: "f32[3072][1]cuda:0", primals_197: "f32[768, 3072][3072, 1]cuda:0", primals_198: "f32[768][1]cuda:0", primals_199: "f32[768][1]cuda:0", primals_200: "f32[768][1]cuda:0", primals_201: "f32[768, 768][768, 1]cuda:0", primals_202: "f32[768][1]cuda:0", primals_203: "f32[768][1]cuda:0", primals_204: "f32[768][1]cuda:0", primals_205: "f32[30522][1]cuda:0", primals_206: "i64[32, 512][512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:96 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.expand.default(primals_3, [1, -1]);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:97 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.gather.default(expand, 1, primals_2);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[32, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(gather, [32, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_4, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_5, expand_1);  primals_5 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:105 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_6, primals_2);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_7)
        add_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[37][1]cuda:0" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:111 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_36: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_1: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_2: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        view: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 768]);  convert_element_type_2 = None
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None
        view_2: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None
        permute_1: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_6: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_7: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_2);  convert_element_type_6 = None
        view_4: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None
        view_5: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [32, 512, -1, 64]);  view_4 = None
        permute_3: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_12: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        permute_4: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_4);  convert_element_type_12 = None
        view_7: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 768]);  addmm_2 = None
        view_8: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [32, 512, -1, 64]);  view_7 = None
        permute_5: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_4: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_5: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_3: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_4, [32, 12, 512, 64]);  mul_4 = None
        clone: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [384, 512, 64]);  clone = None
        expand_4: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_5, [32, 12, 64, 512]);  mul_5 = None
        clone_1: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_10: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [384, 64, 512]);  clone_1 = None
        bmm: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 12, 512, 512])
        convert_element_type_20: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32)
        amax: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_20, [-1], True)
        sub_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax);  convert_element_type_20 = None
        exp: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_11, -inf);  view_11 = None
        logical_not: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_1: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_21);  convert_element_type_21 = None
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_35: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_35: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_35, torch.bfloat16);  inductor_random_default_35 = None
        gt_1: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_35, 0.1);  convert_element_type_default_35 = None
        mul_6: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, where_1);  where_1 = None
        mul_7: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_7, [32, 12, 512, 512]);  mul_7 = None
        view_12: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_5, [384, 512, 512]);  expand_5 = None
        expand_6: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [32, 12, 512, 64]);  permute_5 = None
        clone_2: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [384, 512, 64]);  clone_2 = None
        bmm_1: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 12, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [32, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_24: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        view_16: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [16384, 768]);  view_15 = None
        permute_8: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        addmm_3: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_24, view_16, permute_8);  convert_element_type_24 = None
        view_17: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_34: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_34: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_34, torch.bfloat16);  inductor_random_default_34 = None
        gt_2: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_34, 0.1);  convert_element_type_default_34 = None
        mul_8: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_9: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, mul_3);  mul_9 = mul_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_7, getitem_3);  add_7 = getitem_3 = None
        mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_17)
        add_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_18);  mul_11 = primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_29: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convert_element_type_30: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16)
        view_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [16384, 768]);  convert_element_type_31 = None
        permute_9: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_4: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_29, view_18, permute_9);  convert_element_type_29 = None
        view_19: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_35: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_12: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.5)
        mul_13: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476);  convert_element_type_35 = None
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_10);  mul_12 = add_10 = None
        convert_element_type_36: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_37: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_38: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        view_20: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_36, [16384, 3072]);  convert_element_type_36 = None
        permute_10: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        addmm_5: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_37, view_20, permute_10);  convert_element_type_37 = None
        view_21: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_33: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_3: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_33, 0.1);  convert_element_type_default_33 = None
        mul_15: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_16: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, add_9);  mul_16 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        mul_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, primals_23)
        add_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, primals_24);  mul_18 = primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_42: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convert_element_type_43: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_44: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16)
        view_22: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [16384, 768]);  convert_element_type_44 = None
        permute_11: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        addmm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_42, view_22, permute_11);  convert_element_type_42 = None
        view_23: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None
        view_24: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [32, 512, -1, 64]);  view_23 = None
        permute_12: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_48: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_49: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        permute_13: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_49, [1, 0]);  convert_element_type_49 = None
        addmm_7: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_48, view_22, permute_13);  convert_element_type_48 = None
        view_26: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None
        view_27: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [32, 512, -1, 64]);  view_26 = None
        permute_14: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_54: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_55: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        permute_15: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        addmm_8: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_54, view_22, permute_15);  convert_element_type_54 = None
        view_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 768]);  addmm_8 = None
        view_30: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [32, 512, -1, 64]);  view_29 = None
        permute_16: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_19: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        permute_17: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_20: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_7: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_19, [32, 12, 512, 64]);  mul_19 = None
        clone_4: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_31: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [384, 512, 64]);  clone_4 = None
        expand_8: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_20, [32, 12, 64, 512]);  mul_20 = None
        clone_5: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_32: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [384, 64, 512]);  clone_5 = None
        bmm_2: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 12, 512, 512]);  bmm_2 = None
        convert_element_type_62: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32)
        amax_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_62, [-1], True)
        sub_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, amax_1);  convert_element_type_62 = amax_1 = None
        exp_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_63: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        eq_1: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_33, -inf);  view_33 = None
        logical_not_2: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_1, convert_element_type_63);  logical_not_3 = convert_element_type_63 = None
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_32: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_32: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_32, torch.bfloat16);  inductor_random_default_32 = None
        gt_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_32, 0.1);  convert_element_type_default_32 = None
        mul_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, where_3)
        mul_22: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, 1.1111111111111112);  mul_21 = None
        expand_9: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_22, [32, 12, 512, 512]);  mul_22 = None
        view_34: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_9, [384, 512, 512]);  expand_9 = None
        expand_10: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [32, 12, 512, 64]);  permute_16 = None
        clone_6: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_35: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [384, 512, 64]);  clone_6 = None
        bmm_3: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [32, 12, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [32, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_66: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_67: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        view_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [16384, 768]);  view_37 = None
        permute_19: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        addmm_9: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_66, view_38, permute_19);  convert_element_type_66 = None
        view_39: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_31: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_5: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_31, 0.1);  convert_element_type_default_31 = None
        mul_23: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_24: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, add_13);  mul_24 = add_13 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        mul_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_26: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, primals_33)
        add_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, primals_34);  mul_26 = primals_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_71: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convert_element_type_72: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_73: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16)
        view_40: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 768]);  convert_element_type_73 = None
        permute_20: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_72, [1, 0]);  convert_element_type_72 = None
        addmm_10: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_71, view_40, permute_20);  convert_element_type_71 = None
        view_41: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_77: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_27: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.5)
        mul_28: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.7071067811865476);  convert_element_type_77 = None
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_29: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, add_18);  mul_27 = add_18 = None
        convert_element_type_78: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_79: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_80: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        view_42: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_78, [16384, 3072]);  convert_element_type_78 = None
        permute_21: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        addmm_11: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_79, view_42, permute_21);  convert_element_type_79 = None
        view_43: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_30: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_30: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_30, torch.bfloat16);  inductor_random_default_30 = None
        gt_6: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_30, 0.1);  convert_element_type_default_30 = None
        mul_30: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, add_17);  mul_31 = add_17 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, primals_39)
        add_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, primals_40);  mul_33 = primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_84: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convert_element_type_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_86: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16)
        view_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_86, [16384, 768]);  convert_element_type_86 = None
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        addmm_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_84, view_44, permute_22);  convert_element_type_84 = None
        view_45: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 768]);  addmm_12 = None
        view_46: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [32, 512, -1, 64]);  view_45 = None
        permute_23: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_90: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_91: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_24: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        addmm_13: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_90, view_44, permute_24);  convert_element_type_90 = None
        view_48: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None
        view_49: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [32, 512, -1, 64]);  view_48 = None
        permute_25: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_96: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_97: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        permute_26: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None
        addmm_14: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_96, view_44, permute_26);  convert_element_type_96 = None
        view_51: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 768]);  addmm_14 = None
        view_52: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [32, 512, -1, 64]);  view_51 = None
        permute_27: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_34: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        permute_28: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_35: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_11: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_34, [32, 12, 512, 64]);  mul_34 = None
        clone_8: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_53: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [384, 512, 64]);  clone_8 = None
        expand_12: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_35, [32, 12, 64, 512]);  mul_35 = None
        clone_9: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_54: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [384, 64, 512]);  clone_9 = None
        bmm_4: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 12, 512, 512]);  bmm_4 = None
        convert_element_type_104: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_55, torch.float32)
        amax_2: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_104, [-1], True)
        sub_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, amax_2);  convert_element_type_104 = amax_2 = None
        exp_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_105: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        eq_2: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_55, -inf);  view_55 = None
        logical_not_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_1, convert_element_type_105);  logical_not_5 = convert_element_type_105 = None
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_29: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_29: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_29, torch.bfloat16);  inductor_random_default_29 = None
        gt_7: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_29, 0.1);  convert_element_type_default_29 = None
        mul_36: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, where_5)
        mul_37: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        expand_13: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_37, [32, 12, 512, 512]);  mul_37 = None
        view_56: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_13, [384, 512, 512]);  expand_13 = None
        expand_14: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [32, 12, 512, 64]);  permute_27 = None
        clone_10: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_57: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [384, 512, 64]);  clone_10 = None
        bmm_5: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 12, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [32, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_108: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convert_element_type_109: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        view_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [16384, 768]);  view_59 = None
        permute_30: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        addmm_15: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_108, view_60, permute_30);  convert_element_type_108 = None
        view_61: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_28: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_28: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_28, torch.bfloat16);  inductor_random_default_28 = None
        gt_8: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_28, 0.1);  convert_element_type_default_28 = None
        mul_38: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_39: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, add_21);  mul_39 = add_21 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_8: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
        mul_40: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_41: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, primals_49)
        add_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, primals_50);  mul_41 = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_113: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_114: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_115: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16)
        view_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_115, [16384, 768]);  convert_element_type_115 = None
        permute_31: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        addmm_16: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_113, view_62, permute_31);  convert_element_type_113 = None
        view_63: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_119: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_42: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.5)
        mul_43: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.7071067811865476);  convert_element_type_119 = None
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_44: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, add_26);  mul_42 = add_26 = None
        convert_element_type_120: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_121: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convert_element_type_122: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        view_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [16384, 3072]);  convert_element_type_120 = None
        permute_32: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        addmm_17: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_121, view_64, permute_32);  convert_element_type_121 = None
        view_65: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_27: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_9: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_27, 0.1);  convert_element_type_default_27 = None
        mul_45: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_46: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, add_25);  mul_46 = add_25 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, getitem_13);  add_27 = getitem_13 = None
        mul_47: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_48: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, primals_55)
        add_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, primals_56);  mul_48 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_126: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_127: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_128: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16)
        view_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_128, [16384, 768]);  convert_element_type_128 = None
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_127, [1, 0]);  convert_element_type_127 = None
        addmm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_126, view_66, permute_33);  convert_element_type_126 = None
        view_67: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None
        view_68: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [32, 512, -1, 64]);  view_67 = None
        permute_34: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_132: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_133: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        permute_35: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_133, [1, 0]);  convert_element_type_133 = None
        addmm_19: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_132, view_66, permute_35);  convert_element_type_132 = None
        view_70: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None
        view_71: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [32, 512, -1, 64]);  view_70 = None
        permute_36: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_138: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_139: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        permute_37: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_138, view_66, permute_37);  convert_element_type_138 = None
        view_73: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None
        view_74: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None
        permute_38: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_49: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        permute_39: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_50: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_15: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_49, [32, 12, 512, 64]);  mul_49 = None
        clone_12: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_75: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [384, 512, 64]);  clone_12 = None
        expand_16: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_50, [32, 12, 64, 512]);  mul_50 = None
        clone_13: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_76: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [384, 64, 512]);  clone_13 = None
        bmm_6: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 12, 512, 512]);  bmm_6 = None
        convert_element_type_146: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32)
        amax_3: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_146, [-1], True)
        sub_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, amax_3);  convert_element_type_146 = amax_3 = None
        exp_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_147: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        eq_3: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_77, -inf);  view_77 = None
        logical_not_6: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_1, convert_element_type_147);  logical_not_7 = convert_element_type_147 = None
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_26: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_26: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_26, torch.bfloat16);  inductor_random_default_26 = None
        gt_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_26, 0.1);  convert_element_type_default_26 = None
        mul_51: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, where_7)
        mul_52: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        expand_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_52, [32, 12, 512, 512]);  mul_52 = None
        view_78: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_17, [384, 512, 512]);  expand_17 = None
        expand_18: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [32, 12, 512, 64]);  permute_38 = None
        clone_14: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_79: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [384, 512, 64]);  clone_14 = None
        bmm_7: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_78, view_79)
        view_80: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [32, 12, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [32, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_150: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_151: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        view_82: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [16384, 768]);  view_81 = None
        permute_41: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        addmm_21: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_150, view_82, permute_41);  convert_element_type_150 = None
        view_83: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_25: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_25, torch.bfloat16);  inductor_random_default_25 = None
        gt_11: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_25, 0.1);  convert_element_type_default_25 = None
        mul_53: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_54: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, add_29);  mul_54 = add_29 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_32: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_31, getitem_15);  add_31 = getitem_15 = None
        mul_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_56: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, primals_65)
        add_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, primals_66);  mul_56 = primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_155: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_156: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_157: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16)
        view_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_157, [16384, 768]);  convert_element_type_157 = None
        permute_42: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        addmm_22: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_155, view_84, permute_42);  convert_element_type_155 = None
        view_85: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_161: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_57: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.5)
        mul_58: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.7071067811865476);  convert_element_type_161 = None
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_59: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, add_34);  mul_57 = add_34 = None
        convert_element_type_162: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_163: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_164: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        view_86: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_162, [16384, 3072]);  convert_element_type_162 = None
        permute_43: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_23: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_163, view_86, permute_43);  convert_element_type_163 = None
        view_87: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_24: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_24, torch.bfloat16);  inductor_random_default_24 = None
        gt_12: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_24, 0.1);  convert_element_type_default_24 = None
        mul_60: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_61: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, add_33);  mul_61 = add_33 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_35, getitem_17);  add_35 = getitem_17 = None
        mul_62: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, primals_71)
        add_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, primals_72);  mul_63 = primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_168: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convert_element_type_169: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_170: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16)
        view_88: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_170, [16384, 768]);  convert_element_type_170 = None
        permute_44: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_169, [1, 0]);  convert_element_type_169 = None
        addmm_24: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_168, view_88, permute_44);  convert_element_type_168 = None
        view_89: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 768]);  addmm_24 = None
        view_90: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [32, 512, -1, 64]);  view_89 = None
        permute_45: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_174: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        convert_element_type_175: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        permute_46: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_175, [1, 0]);  convert_element_type_175 = None
        addmm_25: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_174, view_88, permute_46);  convert_element_type_174 = None
        view_92: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None
        view_93: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [32, 512, -1, 64]);  view_92 = None
        permute_47: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_180: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_181: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        permute_48: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_180, view_88, permute_48);  convert_element_type_180 = None
        view_95: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768]);  addmm_26 = None
        view_96: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [32, 512, -1, 64]);  view_95 = None
        permute_49: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_64: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        permute_50: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_65: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_19: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_64, [32, 12, 512, 64]);  mul_64 = None
        clone_16: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_97: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [384, 512, 64]);  clone_16 = None
        expand_20: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_65, [32, 12, 64, 512]);  mul_65 = None
        clone_17: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_98: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [384, 64, 512]);  clone_17 = None
        bmm_8: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 12, 512, 512]);  bmm_8 = None
        convert_element_type_188: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32)
        amax_4: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_188, [-1], True)
        sub_13: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, amax_4);  convert_element_type_188 = amax_4 = None
        exp_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_189: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        eq_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_99, -inf);  view_99 = None
        logical_not_8: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_1, convert_element_type_189);  logical_not_9 = convert_element_type_189 = None
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_23: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_23: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_13: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_23, 0.1);  convert_element_type_default_23 = None
        mul_66: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, where_9)
        mul_67: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        expand_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_67, [32, 12, 512, 512]);  mul_67 = None
        view_100: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_21, [384, 512, 512]);  expand_21 = None
        expand_22: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [32, 12, 512, 64]);  permute_49 = None
        clone_18: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_101: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [384, 512, 64]);  clone_18 = None
        bmm_9: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_100, view_101)
        view_102: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [32, 12, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [32, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_192: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convert_element_type_193: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        view_104: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [16384, 768]);  view_103 = None
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        addmm_27: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_192, view_104, permute_52);  convert_element_type_192 = None
        view_105: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_22: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_22: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_14: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_22, 0.1);  convert_element_type_default_22 = None
        mul_68: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_69: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, add_37);  mul_69 = add_37 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_40: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_14: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_39, getitem_19);  add_39 = getitem_19 = None
        mul_70: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_71: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, primals_81)
        add_41: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, primals_82);  mul_71 = primals_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_197: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convert_element_type_198: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_199: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16)
        view_106: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_199, [16384, 768]);  convert_element_type_199 = None
        permute_53: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        addmm_28: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_197, view_106, permute_53);  convert_element_type_197 = None
        view_107: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_203: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_72: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.5)
        mul_73: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.7071067811865476);  convert_element_type_203 = None
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_74: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, add_42);  mul_72 = add_42 = None
        convert_element_type_204: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_74, torch.bfloat16);  mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_205: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_206: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        view_108: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_204, [16384, 3072]);  convert_element_type_204 = None
        permute_54: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_29: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_205, view_108, permute_54);  convert_element_type_205 = None
        view_109: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_21: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_15: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_21, 0.1);  convert_element_type_default_21 = None
        mul_75: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_76: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, add_41);  mul_76 = add_41 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_43, getitem_21);  add_43 = getitem_21 = None
        mul_77: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_78: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, primals_87)
        add_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, primals_88);  mul_78 = primals_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_210: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_211: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_212: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16)
        view_110: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [16384, 768]);  convert_element_type_212 = None
        permute_55: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None
        addmm_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_210, view_110, permute_55);  convert_element_type_210 = None
        view_111: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 768]);  addmm_30 = None
        view_112: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None
        permute_56: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_216: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_217: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        addmm_31: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_216, view_110, permute_57);  convert_element_type_216 = None
        view_114: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 768]);  addmm_31 = None
        view_115: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [32, 512, -1, 64]);  view_114 = None
        permute_58: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_222: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convert_element_type_223: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        permute_59: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_222, view_110, permute_59);  convert_element_type_222 = None
        view_117: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None
        view_118: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [32, 512, -1, 64]);  view_117 = None
        permute_60: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_79: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        permute_61: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_80: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_23: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_79, [32, 12, 512, 64]);  mul_79 = None
        clone_20: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_119: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [384, 512, 64]);  clone_20 = None
        expand_24: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_80, [32, 12, 64, 512]);  mul_80 = None
        clone_21: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_120: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [384, 64, 512]);  clone_21 = None
        bmm_10: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 12, 512, 512]);  bmm_10 = None
        convert_element_type_230: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.float32)
        amax_5: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_230, [-1], True)
        sub_16: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, amax_5);  convert_element_type_230 = amax_5 = None
        exp_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_231: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        eq_5: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_121, -inf);  view_121 = None
        logical_not_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_1, convert_element_type_231);  logical_not_11 = convert_element_type_231 = None
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_20: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_20: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_20, torch.bfloat16);  inductor_random_default_20 = None
        gt_16: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_20, 0.1);  convert_element_type_default_20 = None
        mul_81: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, where_11)
        mul_82: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, 1.1111111111111112);  mul_81 = None
        expand_25: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_82, [32, 12, 512, 512]);  mul_82 = None
        view_122: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_25, [384, 512, 512]);  expand_25 = None
        expand_26: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [32, 12, 512, 64]);  permute_60 = None
        clone_22: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_123: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [384, 512, 64]);  clone_22 = None
        bmm_11: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [32, 12, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [32, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_234: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convert_element_type_235: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        view_126: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [16384, 768]);  view_125 = None
        permute_63: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        addmm_33: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_234, view_126, permute_63);  convert_element_type_234 = None
        view_127: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_19: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_19: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_17: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_19, 0.1);  convert_element_type_default_19 = None
        mul_83: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_84: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, add_45);  mul_84 = add_45 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_48: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_47, getitem_23);  add_47 = getitem_23 = None
        mul_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_86: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, primals_97)
        add_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, primals_98);  mul_86 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_239: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_240: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_241: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16)
        view_128: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 768]);  convert_element_type_241 = None
        permute_64: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        addmm_34: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_239, view_128, permute_64);  convert_element_type_239 = None
        view_129: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_245: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_87: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.5)
        mul_88: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.7071067811865476);  convert_element_type_245 = None
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_89: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, add_50);  mul_87 = add_50 = None
        convert_element_type_246: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_247: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_248: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        view_130: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_246, [16384, 3072]);  convert_element_type_246 = None
        permute_65: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        addmm_35: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_247, view_130, permute_65);  convert_element_type_247 = None
        view_131: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_18: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_18: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_18, 0.1);  convert_element_type_default_18 = None
        mul_90: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_91: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, add_49);  mul_91 = add_49 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_51, getitem_25);  add_51 = getitem_25 = None
        mul_92: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, primals_103)
        add_53: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, primals_104);  mul_93 = primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_252: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_253: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_254: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16)
        view_132: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_254, [16384, 768]);  convert_element_type_254 = None
        permute_66: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_252, view_132, permute_66);  convert_element_type_252 = None
        view_133: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 768]);  addmm_36 = None
        view_134: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [32, 512, -1, 64]);  view_133 = None
        permute_67: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_258: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_259: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        permute_68: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        addmm_37: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_258, view_132, permute_68);  convert_element_type_258 = None
        view_136: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 768]);  addmm_37 = None
        view_137: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [32, 512, -1, 64]);  view_136 = None
        permute_69: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_264: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_265: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_70: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None
        addmm_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_264, view_132, permute_70);  convert_element_type_264 = None
        view_139: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 768]);  addmm_38 = None
        view_140: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [32, 512, -1, 64]);  view_139 = None
        permute_71: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_94: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_67, 0.3535533905932738);  permute_67 = None
        permute_72: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        mul_95: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_27: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_94, [32, 12, 512, 64]);  mul_94 = None
        clone_24: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_141: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [384, 512, 64]);  clone_24 = None
        expand_28: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_95, [32, 12, 64, 512]);  mul_95 = None
        clone_25: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_142: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [384, 64, 512]);  clone_25 = None
        bmm_12: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 12, 512, 512]);  bmm_12 = None
        convert_element_type_272: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32)
        amax_6: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_272, [-1], True)
        sub_19: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, amax_6);  convert_element_type_272 = amax_6 = None
        exp_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_273: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        eq_6: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_143, -inf);  view_143 = None
        logical_not_12: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_13: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_13, full_default_1, convert_element_type_273);  logical_not_13 = convert_element_type_273 = None
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_17: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_19: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_17, 0.1);  convert_element_type_default_17 = None
        mul_96: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, where_13)
        mul_97: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        expand_29: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_97, [32, 12, 512, 512]);  mul_97 = None
        view_144: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [384, 512, 512]);  expand_29 = None
        expand_30: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_71, [32, 12, 512, 64]);  permute_71 = None
        clone_26: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_145: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [384, 512, 64]);  clone_26 = None
        bmm_13: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_144, view_145)
        view_146: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [32, 12, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [32, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_276: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_277: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        view_148: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [16384, 768]);  view_147 = None
        permute_74: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        addmm_39: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_276, view_148, permute_74);  convert_element_type_276 = None
        view_149: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_16: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_16: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_20: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_16, 0.1);  convert_element_type_default_16 = None
        mul_98: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_99: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, add_53);  mul_99 = add_53 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_56: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_55, getitem_27);  add_55 = getitem_27 = None
        mul_100: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_101: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, primals_113)
        add_57: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, primals_114);  mul_101 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_281: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_282: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_283: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16)
        view_150: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_283, [16384, 768]);  convert_element_type_283 = None
        permute_75: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_282, [1, 0]);  convert_element_type_282 = None
        addmm_40: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_281, view_150, permute_75);  convert_element_type_281 = None
        view_151: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_287: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_102: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.5)
        mul_103: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.7071067811865476);  convert_element_type_287 = None
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_104: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, add_58);  mul_102 = add_58 = None
        convert_element_type_288: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_289: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_290: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        view_152: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_288, [16384, 3072]);  convert_element_type_288 = None
        permute_76: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_290, [1, 0]);  convert_element_type_290 = None
        addmm_41: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_289, view_152, permute_76);  convert_element_type_289 = None
        view_153: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_15: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_21: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_15, 0.1);  convert_element_type_default_15 = None
        mul_105: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_106: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, add_57);  mul_106 = add_57 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_59, getitem_29);  add_59 = getitem_29 = None
        mul_107: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_108: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, primals_119)
        add_61: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_108, primals_120);  mul_108 = primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_294: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_295: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_296: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_154: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_296, [16384, 768]);  convert_element_type_296 = None
        permute_77: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_295, [1, 0]);  convert_element_type_295 = None
        addmm_42: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_294, view_154, permute_77);  convert_element_type_294 = None
        view_155: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 768]);  addmm_42 = None
        view_156: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [32, 512, -1, 64]);  view_155 = None
        permute_78: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_300: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_301: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        permute_79: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        addmm_43: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_300, view_154, permute_79);  convert_element_type_300 = None
        view_158: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 768]);  addmm_43 = None
        view_159: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [32, 512, -1, 64]);  view_158 = None
        permute_80: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_306: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_307: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        permute_81: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_307, [1, 0]);  convert_element_type_307 = None
        addmm_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_306, view_154, permute_81);  convert_element_type_306 = None
        view_161: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 768]);  addmm_44 = None
        view_162: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [32, 512, -1, 64]);  view_161 = None
        permute_82: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_109: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_78, 0.3535533905932738);  permute_78 = None
        permute_83: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        mul_110: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_31: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_109, [32, 12, 512, 64]);  mul_109 = None
        clone_28: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_163: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [384, 512, 64]);  clone_28 = None
        expand_32: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_110, [32, 12, 64, 512]);  mul_110 = None
        clone_29: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_164: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [384, 64, 512]);  clone_29 = None
        bmm_14: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 12, 512, 512]);  bmm_14 = None
        convert_element_type_314: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)
        amax_7: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_314, [-1], True)
        sub_22: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_314, amax_7);  convert_element_type_314 = amax_7 = None
        exp_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_315: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        eq_7: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_165, -inf);  view_165 = None
        logical_not_14: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_15: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_15, full_default_1, convert_element_type_315);  logical_not_15 = convert_element_type_315 = None
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_14: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_14: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_22: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_14, 0.1);  convert_element_type_default_14 = None
        mul_111: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, where_15)
        mul_112: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        expand_33: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_112, [32, 12, 512, 512]);  mul_112 = None
        view_166: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [384, 512, 512]);  expand_33 = None
        expand_34: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [32, 12, 512, 64]);  permute_82 = None
        clone_30: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_167: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [384, 512, 64]);  clone_30 = None
        bmm_15: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_166, view_167)
        view_168: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [32, 12, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [32, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_318: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_319: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        view_170: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [16384, 768]);  view_169 = None
        permute_85: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_319, [1, 0]);  convert_element_type_319 = None
        addmm_45: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_318, view_170, permute_85);  convert_element_type_318 = None
        view_171: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_13: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_23: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_13, 0.1);  convert_element_type_default_13 = None
        mul_113: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_114: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, add_61);  mul_114 = add_61 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_64: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_63, getitem_31);  add_63 = getitem_31 = None
        mul_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_116: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, primals_129)
        add_65: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, primals_130);  mul_116 = primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_323: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_324: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_325: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        view_172: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_325, [16384, 768]);  convert_element_type_325 = None
        permute_86: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        addmm_46: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_323, view_172, permute_86);  convert_element_type_323 = None
        view_173: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_329: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_117: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.5)
        mul_118: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.7071067811865476);  convert_element_type_329 = None
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_119: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, add_66);  mul_117 = add_66 = None
        convert_element_type_330: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_331: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_332: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        view_174: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_330, [16384, 3072]);  convert_element_type_330 = None
        permute_87: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_332, [1, 0]);  convert_element_type_332 = None
        addmm_47: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_331, view_174, permute_87);  convert_element_type_331 = None
        view_175: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_12: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_24: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_12, 0.1);  convert_element_type_default_12 = None
        mul_120: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_121: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, add_65);  mul_121 = add_65 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_67, getitem_33);  add_67 = getitem_33 = None
        mul_122: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, primals_135)
        add_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, primals_136);  mul_123 = primals_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_336: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convert_element_type_337: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_338: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16)
        view_176: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_338, [16384, 768]);  convert_element_type_338 = None
        permute_88: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_337, [1, 0]);  convert_element_type_337 = None
        addmm_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_336, view_176, permute_88);  convert_element_type_336 = None
        view_177: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None
        view_178: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [32, 512, -1, 64]);  view_177 = None
        permute_89: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_342: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convert_element_type_343: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        permute_90: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_343, [1, 0]);  convert_element_type_343 = None
        addmm_49: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_342, view_176, permute_90);  convert_element_type_342 = None
        view_180: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 768]);  addmm_49 = None
        view_181: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [32, 512, -1, 64]);  view_180 = None
        permute_91: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_348: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_349: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        addmm_50: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_348, view_176, permute_92);  convert_element_type_348 = None
        view_183: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 768]);  addmm_50 = None
        view_184: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None
        permute_93: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_124: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_89, 0.3535533905932738);  permute_89 = None
        permute_94: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_125: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_35: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_124, [32, 12, 512, 64]);  mul_124 = None
        clone_32: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_185: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [384, 512, 64]);  clone_32 = None
        expand_36: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_125, [32, 12, 64, 512]);  mul_125 = None
        clone_33: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_186: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [384, 64, 512]);  clone_33 = None
        bmm_16: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 12, 512, 512]);  bmm_16 = None
        convert_element_type_356: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.float32)
        amax_8: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_356, [-1], True)
        sub_25: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_356, amax_8);  convert_element_type_356 = amax_8 = None
        exp_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_357: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        eq_8: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_187, -inf);  view_187 = None
        logical_not_16: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_17, full_default_1, convert_element_type_357);  logical_not_17 = convert_element_type_357 = None
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        convert_element_type_default_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_25: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_11, 0.1);  convert_element_type_default_11 = None
        mul_126: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_25, where_17)
        mul_127: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, 1.1111111111111112);  mul_126 = None
        expand_37: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_127, [32, 12, 512, 512]);  mul_127 = None
        view_188: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [384, 512, 512]);  expand_37 = None
        expand_38: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [32, 12, 512, 64]);  permute_93 = None
        clone_34: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_189: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [384, 512, 64]);  clone_34 = None
        bmm_17: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_188, view_189)
        view_190: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [32, 12, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [32, 512, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_360: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_361: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        view_192: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [16384, 768]);  view_191 = None
        permute_96: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_51: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_360, view_192, permute_96);  convert_element_type_360 = None
        view_193: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_10: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        convert_element_type_default_10: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_26: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_10, 0.1);  convert_element_type_default_10 = None
        mul_128: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_129: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, add_69);  mul_129 = add_69 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_72: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_26: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_71, getitem_35);  add_71 = getitem_35 = None
        mul_130: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_131: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, primals_145)
        add_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, primals_146);  mul_131 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_365: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_366: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_367: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16)
        view_194: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_367, [16384, 768]);  convert_element_type_367 = None
        permute_97: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_366, [1, 0]);  convert_element_type_366 = None
        addmm_52: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_365, view_194, permute_97);  convert_element_type_365 = None
        view_195: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_371: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_132: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, 0.5)
        mul_133: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, 0.7071067811865476);  convert_element_type_371 = None
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_134: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, add_74);  mul_132 = add_74 = None
        convert_element_type_372: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_373: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_374: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        view_196: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_372, [16384, 3072]);  convert_element_type_372 = None
        permute_98: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_374, [1, 0]);  convert_element_type_374 = None
        addmm_53: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_373, view_196, permute_98);  convert_element_type_373 = None
        view_197: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        convert_element_type_default_9: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_27: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_9, 0.1);  convert_element_type_default_9 = None
        mul_135: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_136: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_136, add_73);  mul_136 = add_73 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_75, getitem_37);  add_75 = getitem_37 = None
        mul_137: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_138: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, primals_151)
        add_77: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_138, primals_152);  mul_138 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_378: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        convert_element_type_379: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_380: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16)
        view_198: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_380, [16384, 768]);  convert_element_type_380 = None
        permute_99: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_379, [1, 0]);  convert_element_type_379 = None
        addmm_54: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_378, view_198, permute_99);  convert_element_type_378 = None
        view_199: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 768]);  addmm_54 = None
        view_200: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [32, 512, -1, 64]);  view_199 = None
        permute_100: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_384: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_385: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        permute_101: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_385, [1, 0]);  convert_element_type_385 = None
        addmm_55: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_384, view_198, permute_101);  convert_element_type_384 = None
        view_202: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None
        view_203: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [32, 512, -1, 64]);  view_202 = None
        permute_102: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_390: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convert_element_type_391: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        permute_103: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        addmm_56: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_390, view_198, permute_103);  convert_element_type_390 = None
        view_205: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 768]);  addmm_56 = None
        view_206: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [32, 512, -1, 64]);  view_205 = None
        permute_104: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_139: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_100, 0.3535533905932738);  permute_100 = None
        permute_105: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        mul_140: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_39: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_139, [32, 12, 512, 64]);  mul_139 = None
        clone_36: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_207: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [384, 512, 64]);  clone_36 = None
        expand_40: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_140, [32, 12, 64, 512]);  mul_140 = None
        clone_37: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_208: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [384, 64, 512]);  clone_37 = None
        bmm_18: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 12, 512, 512]);  bmm_18 = None
        convert_element_type_398: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_209, torch.float32)
        amax_9: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_398, [-1], True)
        sub_28: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_398, amax_9);  convert_element_type_398 = amax_9 = None
        exp_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_399: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        eq_9: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_209, -inf);  view_209 = None
        logical_not_18: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_19: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_19, full_default_1, convert_element_type_399);  logical_not_19 = convert_element_type_399 = None
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_8: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_28: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_8, 0.1);  convert_element_type_default_8 = None
        mul_141: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_28, where_19)
        mul_142: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        expand_41: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_142, [32, 12, 512, 512]);  mul_142 = None
        view_210: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [384, 512, 512]);  expand_41 = None
        expand_42: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_104, [32, 12, 512, 64]);  permute_104 = None
        clone_38: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_211: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [384, 512, 64]);  clone_38 = None
        bmm_19: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [32, 12, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [32, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_402: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_403: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        view_214: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [16384, 768]);  view_213 = None
        permute_107: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_403, [1, 0]);  convert_element_type_403 = None
        addmm_57: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_402, view_214, permute_107);  convert_element_type_402 = None
        view_215: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_7: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_7: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_29: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_143: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_144: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, add_77);  mul_144 = add_77 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_80: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_79, getitem_39);  add_79 = getitem_39 = None
        mul_145: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_146: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, primals_161)
        add_81: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, primals_162);  mul_146 = primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_407: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_408: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_409: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16)
        view_216: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [16384, 768]);  convert_element_type_409 = None
        permute_108: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_408, [1, 0]);  convert_element_type_408 = None
        addmm_58: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_407, view_216, permute_108);  convert_element_type_407 = None
        view_217: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_413: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_147: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.5)
        mul_148: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.7071067811865476);  convert_element_type_413 = None
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_149: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, add_82);  mul_147 = add_82 = None
        convert_element_type_414: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_415: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        convert_element_type_416: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        view_218: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_414, [16384, 3072]);  convert_element_type_414 = None
        permute_109: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_416, [1, 0]);  convert_element_type_416 = None
        addmm_59: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_415, view_218, permute_109);  convert_element_type_415 = None
        view_219: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_6: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        convert_element_type_default_6: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_30: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_6, 0.1);  convert_element_type_default_6 = None
        mul_150: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_151: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, add_81);  mul_151 = add_81 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_30: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_83, getitem_41);  add_83 = getitem_41 = None
        mul_152: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_153: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, primals_167)
        add_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, primals_168);  mul_153 = primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_420: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_421: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_422: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_220: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_422, [16384, 768]);  convert_element_type_422 = None
        permute_110: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_421, [1, 0]);  convert_element_type_421 = None
        addmm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_420, view_220, permute_110);  convert_element_type_420 = None
        view_221: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None
        view_222: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64]);  view_221 = None
        permute_111: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_426: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convert_element_type_427: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        permute_112: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_427, [1, 0]);  convert_element_type_427 = None
        addmm_61: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_426, view_220, permute_112);  convert_element_type_426 = None
        view_224: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 768]);  addmm_61 = None
        view_225: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [32, 512, -1, 64]);  view_224 = None
        permute_113: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_432: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        convert_element_type_433: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        permute_114: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_433, [1, 0]);  convert_element_type_433 = None
        addmm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_432, view_220, permute_114);  convert_element_type_432 = None
        view_227: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None
        view_228: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [32, 512, -1, 64]);  view_227 = None
        permute_115: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_154: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_111, 0.3535533905932738);  permute_111 = None
        permute_116: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        mul_155: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_43: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_154, [32, 12, 512, 64]);  mul_154 = None
        clone_40: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_229: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [384, 512, 64]);  clone_40 = None
        expand_44: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_155, [32, 12, 64, 512]);  mul_155 = None
        clone_41: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_230: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [384, 64, 512]);  clone_41 = None
        bmm_20: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 12, 512, 512]);  bmm_20 = None
        convert_element_type_440: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_231, torch.float32)
        amax_10: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_440, [-1], True)
        sub_31: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, amax_10);  convert_element_type_440 = amax_10 = None
        exp_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_441: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        eq_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_231, -inf);  view_231 = None
        logical_not_20: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_21, full_default_1, convert_element_type_441);  logical_not_21 = convert_element_type_441 = None
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        convert_element_type_default_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_31: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_156: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_31, where_21)
        mul_157: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None
        expand_45: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_157, [32, 12, 512, 512]);  mul_157 = None
        view_232: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [384, 512, 512]);  expand_45 = None
        expand_46: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_115, [32, 12, 512, 64]);  permute_115 = None
        clone_42: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_233: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [384, 512, 64]);  clone_42 = None
        bmm_21: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_232, view_233)
        view_234: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [32, 12, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [32, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_444: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convert_element_type_445: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        view_236: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [16384, 768]);  view_235 = None
        permute_118: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_445, [1, 0]);  convert_element_type_445 = None
        addmm_63: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_444, view_236, permute_118);  convert_element_type_444 = None
        view_237: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_4: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_4: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_32: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.1);  convert_element_type_default_4 = None
        mul_158: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_159: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, 1.1111111111111112);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_159, add_85);  mul_159 = add_85 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_88: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_87, getitem_43);  add_87 = getitem_43 = None
        mul_160: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_161: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, primals_177)
        add_89: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, primals_178);  mul_161 = primals_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_449: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        convert_element_type_450: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_451: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16)
        view_238: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_451, [16384, 768]);  convert_element_type_451 = None
        permute_119: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_450, [1, 0]);  convert_element_type_450 = None
        addmm_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_449, view_238, permute_119);  convert_element_type_449 = None
        view_239: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_455: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_162: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.5)
        mul_163: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.7071067811865476);  convert_element_type_455 = None
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_164: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, add_90);  mul_162 = add_90 = None
        convert_element_type_456: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_457: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_458: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        view_240: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_456, [16384, 3072]);  convert_element_type_456 = None
        permute_120: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_458, [1, 0]);  convert_element_type_458 = None
        addmm_65: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_457, view_240, permute_120);  convert_element_type_457 = None
        view_241: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        convert_element_type_default_3: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_33: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_3, 0.1);  convert_element_type_default_3 = None
        mul_165: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_166: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_166, add_89);  mul_166 = add_89 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_91, getitem_45);  add_91 = getitem_45 = None
        mul_167: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_168: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, primals_183)
        add_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_168, primals_184);  mul_168 = primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_462: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        convert_element_type_463: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_464: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        view_242: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_464, [16384, 768]);  convert_element_type_464 = None
        permute_121: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_463, [1, 0]);  convert_element_type_463 = None
        addmm_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_462, view_242, permute_121);  convert_element_type_462 = None
        view_243: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 768]);  addmm_66 = None
        view_244: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [32, 512, -1, 64]);  view_243 = None
        permute_122: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_468: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_469: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        permute_123: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_469, [1, 0]);  convert_element_type_469 = None
        addmm_67: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_468, view_242, permute_123);  convert_element_type_468 = None
        view_246: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        view_247: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [32, 512, -1, 64]);  view_246 = None
        permute_124: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_474: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        convert_element_type_475: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        permute_125: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_475, [1, 0]);  convert_element_type_475 = None
        addmm_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_474, view_242, permute_125);  convert_element_type_474 = None
        view_249: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 768]);  addmm_68 = None
        view_250: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [32, 512, -1, 64]);  view_249 = None
        permute_126: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_169: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_122, 0.3535533905932738);  permute_122 = None
        permute_127: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        mul_170: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_47: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_169, [32, 12, 512, 64]);  mul_169 = None
        clone_44: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_251: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [384, 512, 64]);  clone_44 = None
        expand_48: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_170, [32, 12, 64, 512]);  mul_170 = None
        clone_45: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_252: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [384, 64, 512]);  clone_45 = None
        bmm_22: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_251, view_252)
        view_253: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 12, 512, 512]);  bmm_22 = None
        convert_element_type_482: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_253, torch.float32)
        amax_11: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_482, [-1], True)
        sub_34: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_482, amax_11);  convert_element_type_482 = amax_11 = None
        exp_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_483: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        eq_11: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_253, -inf);  view_253 = None
        logical_not_22: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_23: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_23, full_default_1, convert_element_type_483);  logical_not_23 = full_default_1 = convert_element_type_483 = None
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        convert_element_type_default_2: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_34: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_171: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_34, where_23)
        mul_172: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        expand_49: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_172, [32, 12, 512, 512]);  mul_172 = None
        view_254: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [384, 512, 512]);  expand_49 = None
        expand_50: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_126, [32, 12, 512, 64]);  permute_126 = None
        clone_46: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_255: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [384, 512, 64]);  clone_46 = None
        bmm_23: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [32, 12, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [32, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_486: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        convert_element_type_487: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        view_258: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [16384, 768]);  view_257 = None
        permute_129: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_487, [1, 0]);  convert_element_type_487 = None
        addmm_69: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_486, view_258, permute_129);  convert_element_type_486 = None
        view_259: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_default_1: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_35: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_173: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_174: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, add_93);  mul_174 = add_93 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_96: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_35: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_95, getitem_47);  add_95 = getitem_47 = None
        mul_175: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_176: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, primals_193)
        add_97: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, primals_194);  mul_176 = primals_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_491: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        convert_element_type_492: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        convert_element_type_493: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16)
        view_260: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_493, [16384, 768]);  convert_element_type_493 = None
        permute_130: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_492, [1, 0]);  convert_element_type_492 = None
        addmm_70: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_491, view_260, permute_130);  convert_element_type_491 = None
        view_261: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_497: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_177: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.5)
        mul_178: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.7071067811865476);  convert_element_type_497 = None
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_179: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, add_98);  mul_177 = add_98 = None
        convert_element_type_498: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_499: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convert_element_type_500: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        view_262: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_498, [16384, 3072]);  convert_element_type_498 = None
        permute_131: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_500, [1, 0]);  convert_element_type_500 = None
        addmm_71: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_499, view_262, permute_131);  convert_element_type_499 = None
        view_263: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_default: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_36: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_180: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_181: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, add_97);  mul_181 = add_97 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_36: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_99, getitem_49);  add_99 = getitem_49 = None
        mul_182: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_183: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, primals_199)
        add_101: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, primals_200);  mul_183 = primals_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_504: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16);  primals_202 = None
        convert_element_type_505: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_506: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None
        view_264: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_506, [16384, 768]);  convert_element_type_506 = None
        permute_132: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_505, [1, 0]);  convert_element_type_505 = None
        addmm_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_504, view_264, permute_132);  convert_element_type_504 = None
        view_265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_510: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_184: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, 0.5)
        mul_185: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, 0.7071067811865476);  convert_element_type_510 = None
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = add_102 = None
        convert_element_type_511: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_512: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_511, torch.float32);  convert_element_type_511 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_512, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_103: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_512, getitem_51);  convert_element_type_512 = None
        mul_187: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_188: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, primals_203);  mul_187 = None
        add_104: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, primals_204);  mul_188 = primals_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        convert_element_type_513: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        convert_element_type_514: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_515: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None
        view_266: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_515, [16384, 768]);  convert_element_type_515 = None
        permute_133: "bf16[768, 30522][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_514, [1, 0]);  convert_element_type_514 = None
        constant_pad_nd_default_2: "bf16[768, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_133, [0, 6, 0, 0])
        full_default_32: "bf16[6][1]cuda:0" = torch.ops.aten.full.default([6], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[30528][1]cuda:0" = torch.ops.aten.cat.default([convert_element_type_513, full_default_32]);  convert_element_type_513 = full_default_32 = None
        addmm_default: "bf16[16384, 30528][30528, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_266, constant_pad_nd_default_2);  cat_default = constant_pad_nd_default_2 = None
        slice_tensor: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6);  addmm_default = None
        view_267: "bf16[32, 512, 30522][15630336, 30528, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [32, 512, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_268: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [-1, 30522])
        view_269: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_206, [-1])
        convert_element_type_519: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        amax_12: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_519, [1], True)
        sub_38: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_519, amax_12);  convert_element_type_519 = None
        exp_12: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        convert_element_type_520: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_39, torch.bfloat16);  sub_39 = None
        convert_element_type_521: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_520, torch.float32);  convert_element_type_520 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_24: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_269, full_default_24);  view_269 = full_default_24 = None
        unsqueeze_3: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_521, 1, unsqueeze_3);  convert_element_type_521 = unsqueeze_3 = None
        squeeze: "f32[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[16384][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_25);  neg = full_default_25 = None
        sum_14: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_522: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_12: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_522);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_134: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        permute_138: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_142: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_146: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_150: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_155: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_156: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_157: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_158: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_161: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_166: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_171: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_175: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_179: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_183: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_188: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_189: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_190: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_191: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_194: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_199: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_204: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_208: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_212: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_216: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_221: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_222: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_223: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_224: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_227: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_232: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_237: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_241: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_245: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_249: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_254: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_255: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_256: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_257: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_260: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_265: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_270: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_274: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_278: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_282: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_287: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_288: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_289: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_290: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_293: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_298: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_303: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_307: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_311: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_315: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_320: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_321: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_322: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_323: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_326: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_331: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_336: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_27: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_340: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_344: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_28: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_348: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_353: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_354: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_355: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_356: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_359: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_364: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_369: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_373: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_377: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_381: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_386: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_387: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_388: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_389: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_392: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_397: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_402: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_31: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_406: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_410: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_32: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_414: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_419: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_420: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_421: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_422: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_425: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_430: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_435: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_33: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_439: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_443: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_447: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_452: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_453: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_454: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_455: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_458: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_463: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_468: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_35: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_472: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_476: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_480: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_485: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_486: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_487: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_488: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_491: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_496: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_501: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_37: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        permute_505: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        permute_509: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        permute_513: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_518: "bf16[384, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_519: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_520: "bf16[384, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_521: "bf16[384, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_524: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_529: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_534: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_12, view_267, primals_1, primals_2, primals_7, primals_17, primals_23, primals_33, primals_39, primals_49, primals_55, primals_65, primals_71, primals_81, primals_87, primals_97, primals_103, primals_113, primals_119, primals_129, primals_135, primals_145, primals_151, primals_161, primals_167, primals_177, primals_183, primals_193, primals_199, primals_203, primals_206, gather, mul, gt, view, bmm, amax, sum_1, logical_not_1, gt_1, view_16, gt_2, mul_10, view_18, addmm_4, view_20, gt_3, mul_17, view_22, where_3, gt_4, view_38, gt_5, mul_25, view_40, addmm_10, view_42, gt_6, mul_32, view_44, where_5, gt_7, view_60, gt_8, mul_40, view_62, addmm_16, view_64, gt_9, mul_47, view_66, where_7, gt_10, view_82, gt_11, mul_55, view_84, addmm_22, view_86, gt_12, mul_62, view_88, where_9, gt_13, view_104, gt_14, mul_70, view_106, addmm_28, view_108, gt_15, mul_77, view_110, where_11, gt_16, view_126, gt_17, mul_85, view_128, addmm_34, view_130, gt_18, mul_92, view_132, where_13, gt_19, view_148, gt_20, mul_100, view_150, addmm_40, view_152, gt_21, mul_107, view_154, where_15, gt_22, view_170, gt_23, mul_115, view_172, addmm_46, view_174, gt_24, mul_122, view_176, where_17, gt_25, view_192, gt_26, mul_130, view_194, addmm_52, view_196, gt_27, mul_137, view_198, where_19, gt_28, view_214, gt_29, mul_145, view_216, addmm_58, view_218, gt_30, mul_152, view_220, where_21, gt_31, view_236, gt_32, mul_160, view_238, addmm_64, view_240, gt_33, mul_167, view_242, where_23, gt_34, view_258, gt_35, mul_175, view_260, addmm_70, view_262, gt_36, mul_182, view_264, addmm_72, getitem_51, rsqrt_25, view_266, view_267, amax_12, log, convert_element_type_522, permute_134, permute_138, div_15, permute_142, permute_146, div_16, permute_150, permute_155, permute_156, permute_157, permute_158, permute_161, permute_166, permute_171, div_17, permute_175, permute_179, div_18, permute_183, permute_188, permute_189, permute_190, permute_191, permute_194, permute_199, permute_204, div_19, permute_208, permute_212, div_20, permute_216, permute_221, permute_222, permute_223, permute_224, permute_227, permute_232, permute_237, div_21, permute_241, permute_245, div_22, permute_249, permute_254, permute_255, permute_256, permute_257, permute_260, permute_265, permute_270, div_23, permute_274, permute_278, div_24, permute_282, permute_287, permute_288, permute_289, permute_290, permute_293, permute_298, permute_303, div_25, permute_307, permute_311, div_26, permute_315, permute_320, permute_321, permute_322, permute_323, permute_326, permute_331, permute_336, div_27, permute_340, permute_344, div_28, permute_348, permute_353, permute_354, permute_355, permute_356, permute_359, permute_364, permute_369, div_29, permute_373, permute_377, div_30, permute_381, permute_386, permute_387, permute_388, permute_389, permute_392, permute_397, permute_402, div_31, permute_406, permute_410, div_32, permute_414, permute_419, permute_420, permute_421, permute_422, permute_425, permute_430, permute_435, div_33, permute_439, permute_443, div_34, permute_447, permute_452, permute_453, permute_454, permute_455, permute_458, permute_463, permute_468, div_35, permute_472, permute_476, div_36, permute_480, permute_485, permute_486, permute_487, permute_488, permute_491, permute_496, permute_501, div_37, permute_505, permute_509, div_38, permute_513, permute_518, permute_519, permute_520, permute_521, permute_524, permute_529, permute_534, div_39)
