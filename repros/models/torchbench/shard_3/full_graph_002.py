class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 476][476, 1]cuda:0", primals_2: "i64[4, 476][476, 1]cuda:0", primals_3: "f32[21128, 768][768, 1]cuda:0", primals_4: "i64[4, 476][476, 1]cuda:0", primals_5: "f32[512, 768][768, 1]cuda:0", primals_6: "f32[2, 768][768, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768][1]cuda:0", primals_9: "f32[768, 768][768, 1]cuda:0", primals_10: "f32[768][1]cuda:0", primals_11: "f32[768, 768][768, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768, 768][768, 1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[768, 768][768, 1]cuda:0", primals_16: "f32[768][1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_18: "f32[768][1]cuda:0", primals_19: "f32[3072, 768][768, 1]cuda:0", primals_20: "f32[3072][1]cuda:0", primals_21: "f32[768, 3072][3072, 1]cuda:0", primals_22: "f32[768][1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_24: "f32[768][1]cuda:0", primals_25: "f32[768, 768][768, 1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[768, 768][768, 1]cuda:0", primals_28: "f32[768][1]cuda:0", primals_29: "f32[768, 768][768, 1]cuda:0", primals_30: "f32[768][1]cuda:0", primals_31: "f32[768, 768][768, 1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_35: "f32[3072, 768][768, 1]cuda:0", primals_36: "f32[3072][1]cuda:0", primals_37: "f32[768, 3072][3072, 1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_39: "f32[768][1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_41: "f32[768, 768][768, 1]cuda:0", primals_42: "f32[768][1]cuda:0", primals_43: "f32[768, 768][768, 1]cuda:0", primals_44: "f32[768][1]cuda:0", primals_45: "f32[768, 768][768, 1]cuda:0", primals_46: "f32[768][1]cuda:0", primals_47: "f32[768, 768][768, 1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_50: "f32[768][1]cuda:0", primals_51: "f32[3072, 768][768, 1]cuda:0", primals_52: "f32[3072][1]cuda:0", primals_53: "f32[768, 3072][3072, 1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[768, 768][768, 1]cuda:0", primals_58: "f32[768][1]cuda:0", primals_59: "f32[768, 768][768, 1]cuda:0", primals_60: "f32[768][1]cuda:0", primals_61: "f32[768, 768][768, 1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_63: "f32[768, 768][768, 1]cuda:0", primals_64: "f32[768][1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_66: "f32[768][1]cuda:0", primals_67: "f32[3072, 768][768, 1]cuda:0", primals_68: "f32[3072][1]cuda:0", primals_69: "f32[768, 3072][3072, 1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_72: "f32[768][1]cuda:0", primals_73: "f32[768, 768][768, 1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_75: "f32[768, 768][768, 1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_77: "f32[768, 768][768, 1]cuda:0", primals_78: "f32[768][1]cuda:0", primals_79: "f32[768, 768][768, 1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_81: "f32[768][1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_83: "f32[3072, 768][768, 1]cuda:0", primals_84: "f32[3072][1]cuda:0", primals_85: "f32[768, 3072][3072, 1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_87: "f32[768][1]cuda:0", primals_88: "f32[768][1]cuda:0", primals_89: "f32[768, 768][768, 1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_91: "f32[768, 768][768, 1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_93: "f32[768, 768][768, 1]cuda:0", primals_94: "f32[768][1]cuda:0", primals_95: "f32[768, 768][768, 1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_100: "f32[3072][1]cuda:0", primals_101: "f32[768, 3072][3072, 1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_105: "f32[768, 768][768, 1]cuda:0", primals_106: "f32[768][1]cuda:0", primals_107: "f32[768, 768][768, 1]cuda:0", primals_108: "f32[768][1]cuda:0", primals_109: "f32[768, 768][768, 1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[768, 768][768, 1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_114: "f32[768][1]cuda:0", primals_115: "f32[3072, 768][768, 1]cuda:0", primals_116: "f32[3072][1]cuda:0", primals_117: "f32[768, 3072][3072, 1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_120: "f32[768][1]cuda:0", primals_121: "f32[768, 768][768, 1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_123: "f32[768, 768][768, 1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_125: "f32[768, 768][768, 1]cuda:0", primals_126: "f32[768][1]cuda:0", primals_127: "f32[768, 768][768, 1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_129: "f32[768][1]cuda:0", primals_130: "f32[768][1]cuda:0", primals_131: "f32[3072, 768][768, 1]cuda:0", primals_132: "f32[3072][1]cuda:0", primals_133: "f32[768, 3072][3072, 1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_135: "f32[768][1]cuda:0", primals_136: "f32[768][1]cuda:0", primals_137: "f32[768, 768][768, 1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768, 768][768, 1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[768, 768][768, 1]cuda:0", primals_142: "f32[768][1]cuda:0", primals_143: "f32[768, 768][768, 1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[3072, 768][768, 1]cuda:0", primals_148: "f32[3072][1]cuda:0", primals_149: "f32[768, 3072][3072, 1]cuda:0", primals_150: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768, 768][768, 1]cuda:0", primals_154: "f32[768][1]cuda:0", primals_155: "f32[768, 768][768, 1]cuda:0", primals_156: "f32[768][1]cuda:0", primals_157: "f32[768, 768][768, 1]cuda:0", primals_158: "f32[768][1]cuda:0", primals_159: "f32[768, 768][768, 1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_162: "f32[768][1]cuda:0", primals_163: "f32[3072, 768][768, 1]cuda:0", primals_164: "f32[3072][1]cuda:0", primals_165: "f32[768, 3072][3072, 1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_169: "f32[768, 768][768, 1]cuda:0", primals_170: "f32[768][1]cuda:0", primals_171: "f32[768, 768][768, 1]cuda:0", primals_172: "f32[768][1]cuda:0", primals_173: "f32[768, 768][768, 1]cuda:0", primals_174: "f32[768][1]cuda:0", primals_175: "f32[768, 768][768, 1]cuda:0", primals_176: "f32[768][1]cuda:0", primals_177: "f32[768][1]cuda:0", primals_178: "f32[768][1]cuda:0", primals_179: "f32[3072, 768][768, 1]cuda:0", primals_180: "f32[3072][1]cuda:0", primals_181: "f32[768, 3072][3072, 1]cuda:0", primals_182: "f32[768][1]cuda:0", primals_183: "f32[768][1]cuda:0", primals_184: "f32[768][1]cuda:0", primals_185: "f32[768, 768][768, 1]cuda:0", primals_186: "f32[768][1]cuda:0", primals_187: "f32[768, 768][768, 1]cuda:0", primals_188: "f32[768][1]cuda:0", primals_189: "f32[768, 768][768, 1]cuda:0", primals_190: "f32[768][1]cuda:0", primals_191: "f32[768, 768][768, 1]cuda:0", primals_192: "f32[768][1]cuda:0", primals_193: "f32[768][1]cuda:0", primals_194: "f32[768][1]cuda:0", primals_195: "f32[3072, 768][768, 1]cuda:0", primals_196: "f32[3072][1]cuda:0", primals_197: "f32[768, 3072][3072, 1]cuda:0", primals_198: "f32[768][1]cuda:0", primals_199: "f32[768][1]cuda:0", primals_200: "f32[768][1]cuda:0", primals_201: "f32[768, 768][768, 1]cuda:0", primals_202: "f32[768][1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:502 in forward, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze: "i64[4, 1, 476][476, 476, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_1, 1);  primals_1 = None
        unsqueeze_1: "i64[4, 1, 1, 476][476, 476, 476, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:509 in forward, code: extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        convert_element_type: "f32[4, 1, 1, 476][476, 476, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:510 in forward, code: extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sub: "f32[4, 1, 1, 476][476, 476, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, convert_element_type);  convert_element_type = None
        mul: "f32[4, 1, 1, 476][476, 476, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, -10000.0);  sub = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:230 in forward, code: position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        iota: "i64[476][1]cuda:0" = torch.ops.prims.iota.default(476, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:231 in forward, code: position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        unsqueeze_2: "i64[1, 476][476, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        expand: "i64[4, 476][0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_2, [4, 476])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:236 in forward, code: words_embeddings = self.word_embeddings(input_ids)
        embedding: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_3, primals_4, 0);  primals_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:239 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_5, expand);  primals_5 = expand = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:240 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_2: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_6, primals_2);  primals_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:242 in forward, code: embeddings = words_embeddings + position_embeddings + token_type_embeddings
        add: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:243 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_2: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_2: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, primals_7)
        add_3: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, primals_8);  mul_2 = primals_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_1: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.float16);  primals_10 = None
        convert_element_type_2: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.float16);  primals_9 = None
        convert_element_type_3: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float16)
        view: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [1904, 768]);  convert_element_type_3 = None
        permute: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        addmm: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1, view, permute);  convert_element_type_1 = None
        view_1: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [4, 476, 768]);  addmm = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_7: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.float16);  primals_12 = None
        convert_element_type_8: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.float16);  primals_11 = None
        permute_1: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm_1: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_7, view, permute_1);  convert_element_type_7 = None
        view_3: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [4, 476, 768]);  addmm_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_13: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.float16);  primals_14 = None
        convert_element_type_14: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.float16);  primals_13 = None
        permute_2: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        addmm_2: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_13, view, permute_2);  convert_element_type_13 = None
        view_5: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [4, 476, 768]);  addmm_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_6: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [4, 476, 12, 64]);  view_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_3: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_7: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [4, 476, 12, 64]);  view_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_4: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_8: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [4, 476, 12, 64]);  view_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_5: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_6: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        expand_1: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_3, [4, 12, 476, 64]);  permute_3 = None
        clone_1: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [48, 476, 64]);  clone_1 = None
        expand_2: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_6, [4, 12, 64, 476]);  permute_6 = None
        clone_2: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [48, 64, 476]);  clone_2 = None
        bmm: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [4, 12, 476, 476])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_4: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div, mul);  div = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_11 = torch.ops.prims.prepare_softmax_online.default(add_4, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_72: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_11[0]
        getitem_73: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_11[1];  prepare_softmax_online_default_11 = None
        sub_tensor_11: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, getitem_72);  add_4 = None
        exp_default_11: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_11);  sub_tensor_11 = None
        div_1: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_11, getitem_73);  exp_default_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_21: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        expand_3: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_21, [4, 12, 476, 476]);  convert_element_type_21 = None
        view_12: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [48, 476, 476]);  expand_3 = None
        expand_4: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [4, 12, 476, 64]);  permute_5 = None
        clone_4: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [48, 476, 64]);  clone_4 = None
        bmm_1: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [4, 12, 476, 64]);  bmm_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_7: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_5: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_15: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [4, 476, 768]);  clone_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_24: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.float16);  primals_16 = None
        convert_element_type_25: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.float16);  primals_15 = None
        view_16: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [1904, 768]);  view_15 = None
        permute_8: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        addmm_3: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_24, view_16, permute_8);  convert_element_type_24 = None
        view_17: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [4, 476, 768]);  addmm_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_5: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, add_3);  view_17 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_3: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_4: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, primals_17)
        add_7: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, primals_18);  mul_4 = primals_18 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_29: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.float16);  primals_20 = None
        convert_element_type_30: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.float16);  primals_19 = None
        convert_element_type_31: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float16)
        view_18: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [1904, 768]);  convert_element_type_31 = None
        permute_9: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_4: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_29, view_18, permute_9);  convert_element_type_29 = None
        view_19: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_5: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        div_2: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_19, 1.4142135623730951);  view_19 = None
        erf: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_2);  div_2 = None
        add_8: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1.0);  erf = None
        mul_6: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_35: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.float16);  primals_22 = None
        convert_element_type_36: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.float16);  primals_21 = None
        view_20: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_6, [1904, 3072]);  mul_6 = None
        permute_10: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        addmm_5: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_35, view_20, permute_10);  convert_element_type_35 = None
        view_21: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [4, 476, 768]);  addmm_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_9: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_21, add_7);  view_21 = add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_7: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_8: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, primals_23)
        add_11: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, primals_24);  mul_8 = primals_24 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_40: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.float16);  primals_26 = None
        convert_element_type_41: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.float16);  primals_25 = None
        convert_element_type_42: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float16)
        view_22: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_42, [1904, 768]);  convert_element_type_42 = None
        permute_11: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        addmm_6: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_40, view_22, permute_11);  convert_element_type_40 = None
        view_23: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [4, 476, 768]);  addmm_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_46: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.float16);  primals_28 = None
        convert_element_type_47: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.float16);  primals_27 = None
        permute_12: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        addmm_7: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_46, view_22, permute_12);  convert_element_type_46 = None
        view_25: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [4, 476, 768]);  addmm_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_52: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.float16);  primals_30 = None
        convert_element_type_53: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.float16);  primals_29 = None
        permute_13: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        addmm_8: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_52, view_22, permute_13);  convert_element_type_52 = None
        view_27: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [4, 476, 768]);  addmm_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_28: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [4, 476, 12, 64]);  view_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_14: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_29: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [4, 476, 12, 64]);  view_25 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_15: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_30: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [4, 476, 12, 64]);  view_27 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_16: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_17: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_5: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_14, [4, 12, 476, 64]);  permute_14 = None
        clone_8: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_31: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [48, 476, 64]);  clone_8 = None
        expand_6: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_17, [4, 12, 64, 476]);  permute_17 = None
        clone_9: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_32: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [48, 64, 476]);  clone_9 = None
        bmm_2: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [4, 12, 476, 476]);  bmm_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_3: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_12: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_3, mul);  div_3 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_10 = torch.ops.prims.prepare_softmax_online.default(add_12, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_70: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_10[0]
        getitem_71: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_10[1];  prepare_softmax_online_default_10 = None
        sub_tensor_10: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_12, getitem_70);  add_12 = getitem_70 = None
        exp_default_10: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_10);  sub_tensor_10 = None
        div_4: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_10, getitem_71);  exp_default_10 = getitem_71 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_60: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.float16)
        expand_7: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_60, [4, 12, 476, 476]);  convert_element_type_60 = None
        view_34: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [48, 476, 476]);  expand_7 = None
        expand_8: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [4, 12, 476, 64]);  permute_16 = None
        clone_11: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [48, 476, 64]);  clone_11 = None
        bmm_3: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [4, 12, 476, 64]);  bmm_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_18: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_12: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_37: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [4, 476, 768]);  clone_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_63: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.float16);  primals_32 = None
        convert_element_type_64: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.float16);  primals_31 = None
        view_38: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [1904, 768]);  view_37 = None
        permute_19: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        addmm_9: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_63, view_38, permute_19);  convert_element_type_63 = None
        view_39: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [4, 476, 768]);  addmm_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_13: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_39, add_11);  view_39 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
        mul_9: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_10: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, primals_33)
        add_15: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, primals_34);  mul_10 = primals_34 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_68: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.float16);  primals_36 = None
        convert_element_type_69: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.float16);  primals_35 = None
        convert_element_type_70: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float16)
        view_40: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_70, [1904, 768]);  convert_element_type_70 = None
        permute_20: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        addmm_10: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_68, view_40, permute_20);  convert_element_type_68 = None
        view_41: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_11: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        div_5: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_41, 1.4142135623730951);  view_41 = None
        erf_1: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_5);  div_5 = None
        add_16: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1.0);  erf_1 = None
        mul_12: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, add_16);  mul_11 = add_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_74: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.float16);  primals_38 = None
        convert_element_type_75: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.float16);  primals_37 = None
        view_42: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [1904, 3072]);  mul_12 = None
        permute_21: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        addmm_11: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_74, view_42, permute_21);  convert_element_type_74 = None
        view_43: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [4, 476, 768]);  addmm_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_17: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_43, add_15);  view_43 = add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_13: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
        mul_14: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, primals_39)
        add_19: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, primals_40);  mul_14 = primals_40 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_79: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.float16);  primals_42 = None
        convert_element_type_80: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.float16);  primals_41 = None
        convert_element_type_81: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float16)
        view_44: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_81, [1904, 768]);  convert_element_type_81 = None
        permute_22: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        addmm_12: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_79, view_44, permute_22);  convert_element_type_79 = None
        view_45: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [4, 476, 768]);  addmm_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_85: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.float16);  primals_44 = None
        convert_element_type_86: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.float16);  primals_43 = None
        permute_23: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_86, [1, 0]);  convert_element_type_86 = None
        addmm_13: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_85, view_44, permute_23);  convert_element_type_85 = None
        view_47: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [4, 476, 768]);  addmm_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_91: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.float16);  primals_46 = None
        convert_element_type_92: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.float16);  primals_45 = None
        permute_24: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        addmm_14: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_91, view_44, permute_24);  convert_element_type_91 = None
        view_49: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [4, 476, 768]);  addmm_14 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_50: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [4, 476, 12, 64]);  view_45 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_25: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_51: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_47, [4, 476, 12, 64]);  view_47 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_26: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_52: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [4, 476, 12, 64]);  view_49 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_27: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_28: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_9: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_25, [4, 12, 476, 64]);  permute_25 = None
        clone_15: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_53: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [48, 476, 64]);  clone_15 = None
        expand_10: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_28, [4, 12, 64, 476]);  permute_28 = None
        clone_16: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_54: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [48, 64, 476]);  clone_16 = None
        bmm_4: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [4, 12, 476, 476]);  bmm_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_6: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_20: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_9 = torch.ops.prims.prepare_softmax_online.default(add_20, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_68: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_9[0]
        getitem_69: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_9[1];  prepare_softmax_online_default_9 = None
        sub_tensor_9: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_20, getitem_68);  add_20 = getitem_68 = None
        exp_default_9: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_9);  sub_tensor_9 = None
        div_7: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_9, getitem_69);  exp_default_9 = getitem_69 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_99: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.float16)
        expand_11: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_99, [4, 12, 476, 476]);  convert_element_type_99 = None
        view_56: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [48, 476, 476]);  expand_11 = None
        expand_12: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [4, 12, 476, 64]);  permute_27 = None
        clone_18: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_57: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [48, 476, 64]);  clone_18 = None
        bmm_5: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [4, 12, 476, 64]);  bmm_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_29: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_19: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_59: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [4, 476, 768]);  clone_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_102: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.float16);  primals_48 = None
        convert_element_type_103: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.float16);  primals_47 = None
        view_60: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [1904, 768]);  view_59 = None
        permute_30: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        addmm_15: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_102, view_60, permute_30);  convert_element_type_102 = None
        view_61: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [4, 476, 768]);  addmm_15 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_21: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_61, add_19);  view_61 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_22: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
        mul_15: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
        mul_16: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, primals_49)
        add_23: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, primals_50);  mul_16 = primals_50 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_107: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.float16);  primals_52 = None
        convert_element_type_108: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.float16);  primals_51 = None
        convert_element_type_109: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float16)
        view_62: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_109, [1904, 768]);  convert_element_type_109 = None
        permute_31: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_16: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_107, view_62, permute_31);  convert_element_type_107 = None
        view_63: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_17: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        div_8: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_63, 1.4142135623730951);  view_63 = None
        erf_2: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_8);  div_8 = None
        add_24: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1.0);  erf_2 = None
        mul_18: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, add_24);  mul_17 = add_24 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_113: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.float16);  primals_54 = None
        convert_element_type_114: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.float16);  primals_53 = None
        view_64: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_18, [1904, 3072]);  mul_18 = None
        permute_32: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        addmm_17: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_113, view_64, permute_32);  convert_element_type_113 = None
        view_65: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [4, 476, 768]);  addmm_17 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_65, add_23);  view_65 = add_23 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_19: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
        mul_20: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, primals_55)
        add_27: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, primals_56);  mul_20 = primals_56 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_118: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.float16);  primals_58 = None
        convert_element_type_119: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.float16);  primals_57 = None
        convert_element_type_120: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float16)
        view_66: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [1904, 768]);  convert_element_type_120 = None
        permute_33: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        addmm_18: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_118, view_66, permute_33);  convert_element_type_118 = None
        view_67: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [4, 476, 768]);  addmm_18 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_124: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.float16);  primals_60 = None
        convert_element_type_125: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.float16);  primals_59 = None
        permute_34: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        addmm_19: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_124, view_66, permute_34);  convert_element_type_124 = None
        view_69: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [4, 476, 768]);  addmm_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_130: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.float16);  primals_62 = None
        convert_element_type_131: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.float16);  primals_61 = None
        permute_35: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        addmm_20: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_130, view_66, permute_35);  convert_element_type_130 = None
        view_71: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [4, 476, 768]);  addmm_20 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_72: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [4, 476, 12, 64]);  view_67 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_36: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_73: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [4, 476, 12, 64]);  view_69 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_37: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_74: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [4, 476, 12, 64]);  view_71 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_38: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_39: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_13: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_36, [4, 12, 476, 64]);  permute_36 = None
        clone_22: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_75: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [48, 476, 64]);  clone_22 = None
        expand_14: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_39, [4, 12, 64, 476]);  permute_39 = None
        clone_23: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_76: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [48, 64, 476]);  clone_23 = None
        bmm_6: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [4, 12, 476, 476]);  bmm_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_9: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_28: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_9, mul);  div_9 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_8 = torch.ops.prims.prepare_softmax_online.default(add_28, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_66: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_8[0]
        getitem_67: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_8[1];  prepare_softmax_online_default_8 = None
        sub_tensor_8: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_28, getitem_66);  add_28 = getitem_66 = None
        exp_default_8: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_8);  sub_tensor_8 = None
        div_10: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_8, getitem_67);  exp_default_8 = getitem_67 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_138: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.float16)
        expand_15: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_138, [4, 12, 476, 476]);  convert_element_type_138 = None
        view_78: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [48, 476, 476]);  expand_15 = None
        expand_16: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [4, 12, 476, 64]);  permute_38 = None
        clone_25: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_79: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [48, 476, 64]);  clone_25 = None
        bmm_7: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_78, view_79)
        view_80: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [4, 12, 476, 64]);  bmm_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_40: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_26: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_81: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [4, 476, 768]);  clone_26 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_141: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.float16);  primals_64 = None
        convert_element_type_142: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.float16);  primals_63 = None
        view_82: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [1904, 768]);  view_81 = None
        permute_41: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0]);  convert_element_type_142 = None
        addmm_21: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_141, view_82, permute_41);  convert_element_type_141 = None
        view_83: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [4, 476, 768]);  addmm_21 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_29: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, add_27);  view_83 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_30: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_21: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
        mul_22: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, primals_65)
        add_31: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, primals_66);  mul_22 = primals_66 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_146: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.float16);  primals_68 = None
        convert_element_type_147: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.float16);  primals_67 = None
        convert_element_type_148: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.float16)
        view_84: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_148, [1904, 768]);  convert_element_type_148 = None
        permute_42: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        addmm_22: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_146, view_84, permute_42);  convert_element_type_146 = None
        view_85: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_23: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        div_11: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_85, 1.4142135623730951);  view_85 = None
        erf_3: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_11);  div_11 = None
        add_32: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1.0);  erf_3 = None
        mul_24: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, add_32);  mul_23 = add_32 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_152: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.float16);  primals_70 = None
        convert_element_type_153: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.float16);  primals_69 = None
        view_86: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_24, [1904, 3072]);  mul_24 = None
        permute_43: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        addmm_23: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_152, view_86, permute_43);  convert_element_type_152 = None
        view_87: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [4, 476, 768]);  addmm_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_33: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_87, add_31);  view_87 = add_31 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_25: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
        mul_26: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, primals_71)
        add_35: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, primals_72);  mul_26 = primals_72 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_157: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.float16);  primals_74 = None
        convert_element_type_158: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.float16);  primals_73 = None
        convert_element_type_159: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.float16)
        view_88: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_159, [1904, 768]);  convert_element_type_159 = None
        permute_44: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_158, [1, 0]);  convert_element_type_158 = None
        addmm_24: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_157, view_88, permute_44);  convert_element_type_157 = None
        view_89: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [4, 476, 768]);  addmm_24 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_163: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.float16);  primals_76 = None
        convert_element_type_164: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.float16);  primals_75 = None
        permute_45: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_25: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_163, view_88, permute_45);  convert_element_type_163 = None
        view_91: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [4, 476, 768]);  addmm_25 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_169: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.float16);  primals_78 = None
        convert_element_type_170: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.float16);  primals_77 = None
        permute_46: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_170, [1, 0]);  convert_element_type_170 = None
        addmm_26: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_169, view_88, permute_46);  convert_element_type_169 = None
        view_93: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [4, 476, 768]);  addmm_26 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_94: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [4, 476, 12, 64]);  view_89 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_47: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_95: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [4, 476, 12, 64]);  view_91 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_48: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_96: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_93, [4, 476, 12, 64]);  view_93 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_49: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_50: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_17: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_47, [4, 12, 476, 64]);  permute_47 = None
        clone_29: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_97: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [48, 476, 64]);  clone_29 = None
        expand_18: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_50, [4, 12, 64, 476]);  permute_50 = None
        clone_30: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_98: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [48, 64, 476]);  clone_30 = None
        bmm_8: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [4, 12, 476, 476]);  bmm_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_12: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_36: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_7 = torch.ops.prims.prepare_softmax_online.default(add_36, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_64: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_7[0]
        getitem_65: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_7[1];  prepare_softmax_online_default_7 = None
        sub_tensor_7: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_36, getitem_64);  add_36 = getitem_64 = None
        exp_default_7: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_7);  sub_tensor_7 = None
        div_13: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_7, getitem_65);  exp_default_7 = getitem_65 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_177: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.float16)
        expand_19: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_177, [4, 12, 476, 476]);  convert_element_type_177 = None
        view_100: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [48, 476, 476]);  expand_19 = None
        expand_20: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [4, 12, 476, 64]);  permute_49 = None
        clone_32: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_101: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [48, 476, 64]);  clone_32 = None
        bmm_9: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_100, view_101)
        view_102: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [4, 12, 476, 64]);  bmm_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_51: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_33: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_103: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [4, 476, 768]);  clone_33 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_180: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.float16);  primals_80 = None
        convert_element_type_181: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.float16);  primals_79 = None
        view_104: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [1904, 768]);  view_103 = None
        permute_52: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_27: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_180, view_104, permute_52);  convert_element_type_180 = None
        view_105: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [4, 476, 768]);  addmm_27 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_37: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_105, add_35);  view_105 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_38: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
        mul_27: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_28: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, primals_81)
        add_39: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, primals_82);  mul_28 = primals_82 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_185: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.float16);  primals_84 = None
        convert_element_type_186: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.float16);  primals_83 = None
        convert_element_type_187: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float16)
        view_106: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_187, [1904, 768]);  convert_element_type_187 = None
        permute_53: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_186, [1, 0]);  convert_element_type_186 = None
        addmm_28: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_185, view_106, permute_53);  convert_element_type_185 = None
        view_107: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_29: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        div_14: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_107, 1.4142135623730951);  view_107 = None
        erf_4: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_14);  div_14 = None
        add_40: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1.0);  erf_4 = None
        mul_30: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, add_40);  mul_29 = add_40 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_191: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.float16);  primals_86 = None
        convert_element_type_192: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.float16);  primals_85 = None
        view_108: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_30, [1904, 3072]);  mul_30 = None
        permute_54: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        addmm_29: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_191, view_108, permute_54);  convert_element_type_191 = None
        view_109: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [4, 476, 768]);  addmm_29 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_41: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, add_39);  view_109 = add_39 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_31: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_32: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, primals_87)
        add_43: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, primals_88);  mul_32 = primals_88 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_196: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.float16);  primals_90 = None
        convert_element_type_197: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.float16);  primals_89 = None
        convert_element_type_198: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float16)
        view_110: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_198, [1904, 768]);  convert_element_type_198 = None
        permute_55: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        addmm_30: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_196, view_110, permute_55);  convert_element_type_196 = None
        view_111: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [4, 476, 768]);  addmm_30 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_202: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.float16);  primals_92 = None
        convert_element_type_203: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.float16);  primals_91 = None
        permute_56: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_203, [1, 0]);  convert_element_type_203 = None
        addmm_31: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_202, view_110, permute_56);  convert_element_type_202 = None
        view_113: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [4, 476, 768]);  addmm_31 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_208: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.float16);  primals_94 = None
        convert_element_type_209: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.float16);  primals_93 = None
        permute_57: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_209, [1, 0]);  convert_element_type_209 = None
        addmm_32: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_208, view_110, permute_57);  convert_element_type_208 = None
        view_115: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [4, 476, 768]);  addmm_32 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_116: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [4, 476, 12, 64]);  view_111 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_58: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_117: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [4, 476, 12, 64]);  view_113 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_59: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_118: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [4, 476, 12, 64]);  view_115 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_60: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_61: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_21: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_58, [4, 12, 476, 64]);  permute_58 = None
        clone_36: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_119: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [48, 476, 64]);  clone_36 = None
        expand_22: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_61, [4, 12, 64, 476]);  permute_61 = None
        clone_37: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_120: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [48, 64, 476]);  clone_37 = None
        bmm_10: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [4, 12, 476, 476]);  bmm_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_15: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_44: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_15, mul);  div_15 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_6 = torch.ops.prims.prepare_softmax_online.default(add_44, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_62: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_6[0]
        getitem_63: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_6[1];  prepare_softmax_online_default_6 = None
        sub_tensor_6: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_44, getitem_62);  add_44 = getitem_62 = None
        exp_default_6: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_6);  sub_tensor_6 = None
        div_16: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_6, getitem_63);  exp_default_6 = getitem_63 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_216: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.float16)
        expand_23: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_216, [4, 12, 476, 476]);  convert_element_type_216 = None
        view_122: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [48, 476, 476]);  expand_23 = None
        expand_24: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [4, 12, 476, 64]);  permute_60 = None
        clone_39: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_123: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [48, 476, 64]);  clone_39 = None
        bmm_11: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [4, 12, 476, 64]);  bmm_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_62: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_40: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_125: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [4, 476, 768]);  clone_40 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_219: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.float16);  primals_96 = None
        convert_element_type_220: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.float16);  primals_95 = None
        view_126: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [1904, 768]);  view_125 = None
        permute_63: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_220, [1, 0]);  convert_element_type_220 = None
        addmm_33: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_219, view_126, permute_63);  convert_element_type_219 = None
        view_127: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [4, 476, 768]);  addmm_33 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_127, add_43);  view_127 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_46: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
        mul_33: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
        mul_34: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, primals_97)
        add_47: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, primals_98);  mul_34 = primals_98 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_224: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.float16);  primals_100 = None
        convert_element_type_225: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.float16);  primals_99 = None
        convert_element_type_226: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float16)
        view_128: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_226, [1904, 768]);  convert_element_type_226 = None
        permute_64: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_225, [1, 0]);  convert_element_type_225 = None
        addmm_34: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_224, view_128, permute_64);  convert_element_type_224 = None
        view_129: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_35: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        div_17: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_129, 1.4142135623730951);  view_129 = None
        erf_5: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_17);  div_17 = None
        add_48: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1.0);  erf_5 = None
        mul_36: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, add_48);  mul_35 = add_48 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_230: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.float16);  primals_102 = None
        convert_element_type_231: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.float16);  primals_101 = None
        view_130: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_36, [1904, 3072]);  mul_36 = None
        permute_65: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        addmm_35: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_230, view_130, permute_65);  convert_element_type_230 = None
        view_131: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [4, 476, 768]);  addmm_35 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_131, add_47);  view_131 = add_47 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_37: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
        mul_38: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, primals_103)
        add_51: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_38, primals_104);  mul_38 = primals_104 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_235: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.float16);  primals_106 = None
        convert_element_type_236: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.float16);  primals_105 = None
        convert_element_type_237: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float16)
        view_132: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_237, [1904, 768]);  convert_element_type_237 = None
        permute_66: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        addmm_36: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_235, view_132, permute_66);  convert_element_type_235 = None
        view_133: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [4, 476, 768]);  addmm_36 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_241: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.float16);  primals_108 = None
        convert_element_type_242: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.float16);  primals_107 = None
        permute_67: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_242, [1, 0]);  convert_element_type_242 = None
        addmm_37: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_241, view_132, permute_67);  convert_element_type_241 = None
        view_135: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [4, 476, 768]);  addmm_37 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_247: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.float16);  primals_110 = None
        convert_element_type_248: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.float16);  primals_109 = None
        permute_68: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        addmm_38: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_247, view_132, permute_68);  convert_element_type_247 = None
        view_137: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [4, 476, 768]);  addmm_38 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_138: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [4, 476, 12, 64]);  view_133 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_69: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_139: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [4, 476, 12, 64]);  view_135 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_70: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_140: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [4, 476, 12, 64]);  view_137 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_71: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_72: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_25: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_69, [4, 12, 476, 64]);  permute_69 = None
        clone_43: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_141: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [48, 476, 64]);  clone_43 = None
        expand_26: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_72, [4, 12, 64, 476]);  permute_72 = None
        clone_44: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_142: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [48, 64, 476]);  clone_44 = None
        bmm_12: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [4, 12, 476, 476]);  bmm_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_18: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_52: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_5 = torch.ops.prims.prepare_softmax_online.default(add_52, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_60: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_5[0]
        getitem_61: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_5[1];  prepare_softmax_online_default_5 = None
        sub_tensor_5: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_52, getitem_60);  add_52 = getitem_60 = None
        exp_default_5: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_5);  sub_tensor_5 = None
        div_19: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_5, getitem_61);  exp_default_5 = getitem_61 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_255: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.float16)
        expand_27: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_255, [4, 12, 476, 476]);  convert_element_type_255 = None
        view_144: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_27, [48, 476, 476]);  expand_27 = None
        expand_28: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_71, [4, 12, 476, 64]);  permute_71 = None
        clone_46: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_145: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [48, 476, 64]);  clone_46 = None
        bmm_13: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_144, view_145)
        view_146: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [4, 12, 476, 64]);  bmm_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_73: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_47: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_147: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [4, 476, 768]);  clone_47 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_258: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.float16);  primals_112 = None
        convert_element_type_259: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.float16);  primals_111 = None
        view_148: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [1904, 768]);  view_147 = None
        permute_74: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        addmm_39: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_258, view_148, permute_74);  convert_element_type_258 = None
        view_149: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [4, 476, 768]);  addmm_39 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_149, add_51);  view_149 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_54: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
        mul_39: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
        mul_40: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, primals_113)
        add_55: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, primals_114);  mul_40 = primals_114 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_263: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.float16);  primals_116 = None
        convert_element_type_264: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.float16);  primals_115 = None
        convert_element_type_265: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float16)
        view_150: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [1904, 768]);  convert_element_type_265 = None
        permute_75: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_40: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_263, view_150, permute_75);  convert_element_type_263 = None
        view_151: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_41: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        div_20: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_151, 1.4142135623730951);  view_151 = None
        erf_6: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_20);  div_20 = None
        add_56: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1.0);  erf_6 = None
        mul_42: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, add_56);  mul_41 = add_56 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_269: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.float16);  primals_118 = None
        convert_element_type_270: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.float16);  primals_117 = None
        view_152: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_42, [1904, 3072]);  mul_42 = None
        permute_76: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_41: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_269, view_152, permute_76);  convert_element_type_269 = None
        view_153: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [4, 476, 768]);  addmm_41 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, add_55);  view_153 = add_55 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_43: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
        mul_44: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, primals_119)
        add_59: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, primals_120);  mul_44 = primals_120 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_274: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.float16);  primals_122 = None
        convert_element_type_275: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.float16);  primals_121 = None
        convert_element_type_276: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float16)
        view_154: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_276, [1904, 768]);  convert_element_type_276 = None
        permute_77: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        addmm_42: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_274, view_154, permute_77);  convert_element_type_274 = None
        view_155: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [4, 476, 768]);  addmm_42 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_280: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.float16);  primals_124 = None
        convert_element_type_281: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.float16);  primals_123 = None
        permute_78: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        addmm_43: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_280, view_154, permute_78);  convert_element_type_280 = None
        view_157: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [4, 476, 768]);  addmm_43 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_286: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.float16);  primals_126 = None
        convert_element_type_287: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.float16);  primals_125 = None
        permute_79: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_44: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_286, view_154, permute_79);  convert_element_type_286 = None
        view_159: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [4, 476, 768]);  addmm_44 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_160: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [4, 476, 12, 64]);  view_155 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_80: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_161: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_157, [4, 476, 12, 64]);  view_157 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_81: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_162: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_159, [4, 476, 12, 64]);  view_159 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_82: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_83: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        expand_29: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_80, [4, 12, 476, 64]);  permute_80 = None
        clone_50: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_163: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [48, 476, 64]);  clone_50 = None
        expand_30: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_83, [4, 12, 64, 476]);  permute_83 = None
        clone_51: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_164: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [48, 64, 476]);  clone_51 = None
        bmm_14: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [4, 12, 476, 476]);  bmm_14 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_21: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_60: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_21, mul);  div_21 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_4 = torch.ops.prims.prepare_softmax_online.default(add_60, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_58: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_4[0]
        getitem_59: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_4[1];  prepare_softmax_online_default_4 = None
        sub_tensor_4: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_60, getitem_58);  add_60 = getitem_58 = None
        exp_default_4: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
        div_22: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_4, getitem_59);  exp_default_4 = getitem_59 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_294: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.float16)
        expand_31: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_294, [4, 12, 476, 476]);  convert_element_type_294 = None
        view_166: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_31, [48, 476, 476]);  expand_31 = None
        expand_32: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [4, 12, 476, 64]);  permute_82 = None
        clone_53: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_167: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [48, 476, 64]);  clone_53 = None
        bmm_15: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_166, view_167)
        view_168: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [4, 12, 476, 64]);  bmm_15 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_84: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_54: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_169: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [4, 476, 768]);  clone_54 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_297: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.float16);  primals_128 = None
        convert_element_type_298: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.float16);  primals_127 = None
        view_170: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [1904, 768]);  view_169 = None
        permute_85: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_298, [1, 0]);  convert_element_type_298 = None
        addmm_45: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_297, view_170, permute_85);  convert_element_type_297 = None
        view_171: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [4, 476, 768]);  addmm_45 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_61: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_171, add_59);  view_171 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_62: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
        mul_45: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
        mul_46: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, primals_129)
        add_63: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, primals_130);  mul_46 = primals_130 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_302: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.float16);  primals_132 = None
        convert_element_type_303: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.float16);  primals_131 = None
        convert_element_type_304: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float16)
        view_172: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [1904, 768]);  convert_element_type_304 = None
        permute_86: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_303, [1, 0]);  convert_element_type_303 = None
        addmm_46: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_302, view_172, permute_86);  convert_element_type_302 = None
        view_173: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_47: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        div_23: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_173, 1.4142135623730951);  view_173 = None
        erf_7: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_23);  div_23 = None
        add_64: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1.0);  erf_7 = None
        mul_48: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, add_64);  mul_47 = add_64 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_308: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.float16);  primals_134 = None
        convert_element_type_309: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.float16);  primals_133 = None
        view_174: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [1904, 3072]);  mul_48 = None
        permute_87: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0]);  convert_element_type_309 = None
        addmm_47: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_308, view_174, permute_87);  convert_element_type_308 = None
        view_175: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [4, 476, 768]);  addmm_47 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_65: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_175, add_63);  view_175 = add_63 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_49: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
        mul_50: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, primals_135)
        add_67: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, primals_136);  mul_50 = primals_136 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_313: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.float16);  primals_138 = None
        convert_element_type_314: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.float16);  primals_137 = None
        convert_element_type_315: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.float16)
        view_176: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_315, [1904, 768]);  convert_element_type_315 = None
        permute_88: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_314, [1, 0]);  convert_element_type_314 = None
        addmm_48: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_313, view_176, permute_88);  convert_element_type_313 = None
        view_177: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [4, 476, 768]);  addmm_48 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_319: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.float16);  primals_140 = None
        convert_element_type_320: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.float16);  primals_139 = None
        permute_89: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_320, [1, 0]);  convert_element_type_320 = None
        addmm_49: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_319, view_176, permute_89);  convert_element_type_319 = None
        view_179: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [4, 476, 768]);  addmm_49 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_325: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.float16);  primals_142 = None
        convert_element_type_326: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.float16);  primals_141 = None
        permute_90: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_326, [1, 0]);  convert_element_type_326 = None
        addmm_50: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_325, view_176, permute_90);  convert_element_type_325 = None
        view_181: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [4, 476, 768]);  addmm_50 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_182: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [4, 476, 12, 64]);  view_177 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_91: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_183: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_179, [4, 476, 12, 64]);  view_179 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_92: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_184: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [4, 476, 12, 64]);  view_181 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_93: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_94: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        expand_33: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_91, [4, 12, 476, 64]);  permute_91 = None
        clone_57: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_185: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [48, 476, 64]);  clone_57 = None
        expand_34: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_94, [4, 12, 64, 476]);  permute_94 = None
        clone_58: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_186: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [48, 64, 476]);  clone_58 = None
        bmm_16: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [4, 12, 476, 476]);  bmm_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_24: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_68: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_24, mul);  div_24 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_3 = torch.ops.prims.prepare_softmax_online.default(add_68, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_56: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_3[0]
        getitem_57: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_3[1];  prepare_softmax_online_default_3 = None
        sub_tensor_3: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_68, getitem_56);  add_68 = getitem_56 = None
        exp_default_3: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_3);  sub_tensor_3 = None
        div_25: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_3, getitem_57);  exp_default_3 = getitem_57 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_333: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.float16)
        expand_35: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_333, [4, 12, 476, 476]);  convert_element_type_333 = None
        view_188: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_35, [48, 476, 476]);  expand_35 = None
        expand_36: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [4, 12, 476, 64]);  permute_93 = None
        clone_60: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_189: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [48, 476, 64]);  clone_60 = None
        bmm_17: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_188, view_189)
        view_190: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [4, 12, 476, 64]);  bmm_17 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_95: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_61: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_191: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [4, 476, 768]);  clone_61 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_336: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.float16);  primals_144 = None
        convert_element_type_337: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.float16);  primals_143 = None
        view_192: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [1904, 768]);  view_191 = None
        permute_96: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_337, [1, 0]);  convert_element_type_337 = None
        addmm_51: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_336, view_192, permute_96);  convert_element_type_336 = None
        view_193: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [4, 476, 768]);  addmm_51 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_69: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, add_67);  view_193 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_70: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
        mul_51: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
        mul_52: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, primals_145)
        add_71: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, primals_146);  mul_52 = primals_146 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_341: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.float16);  primals_148 = None
        convert_element_type_342: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.float16);  primals_147 = None
        convert_element_type_343: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float16)
        view_194: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_343, [1904, 768]);  convert_element_type_343 = None
        permute_97: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        addmm_52: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_341, view_194, permute_97);  convert_element_type_341 = None
        view_195: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_53: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        div_26: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_195, 1.4142135623730951);  view_195 = None
        erf_8: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_26);  div_26 = None
        add_72: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1.0);  erf_8 = None
        mul_54: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, add_72);  mul_53 = add_72 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_347: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.float16);  primals_150 = None
        convert_element_type_348: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.float16);  primals_149 = None
        view_196: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [1904, 3072]);  mul_54 = None
        permute_98: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_348, [1, 0]);  convert_element_type_348 = None
        addmm_53: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_347, view_196, permute_98);  convert_element_type_347 = None
        view_197: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [4, 476, 768]);  addmm_53 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_197, add_71);  view_197 = add_71 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_55: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
        mul_56: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, primals_151)
        add_75: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, primals_152);  mul_56 = primals_152 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_352: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.float16);  primals_154 = None
        convert_element_type_353: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.float16);  primals_153 = None
        convert_element_type_354: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.float16)
        view_198: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_354, [1904, 768]);  convert_element_type_354 = None
        permute_99: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_353, [1, 0]);  convert_element_type_353 = None
        addmm_54: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_352, view_198, permute_99);  convert_element_type_352 = None
        view_199: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [4, 476, 768]);  addmm_54 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_358: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.float16);  primals_156 = None
        convert_element_type_359: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.float16);  primals_155 = None
        permute_100: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_359, [1, 0]);  convert_element_type_359 = None
        addmm_55: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_358, view_198, permute_100);  convert_element_type_358 = None
        view_201: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [4, 476, 768]);  addmm_55 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_364: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.float16);  primals_158 = None
        convert_element_type_365: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.float16);  primals_157 = None
        permute_101: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_365, [1, 0]);  convert_element_type_365 = None
        addmm_56: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_364, view_198, permute_101);  convert_element_type_364 = None
        view_203: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [4, 476, 768]);  addmm_56 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_204: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [4, 476, 12, 64]);  view_199 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_102: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_205: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [4, 476, 12, 64]);  view_201 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_103: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_206: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_203, [4, 476, 12, 64]);  view_203 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_104: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_105: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        expand_37: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_102, [4, 12, 476, 64]);  permute_102 = None
        clone_64: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_207: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [48, 476, 64]);  clone_64 = None
        expand_38: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_105, [4, 12, 64, 476]);  permute_105 = None
        clone_65: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_208: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [48, 64, 476]);  clone_65 = None
        bmm_18: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [4, 12, 476, 476]);  bmm_18 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_27: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_76: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_27, mul);  div_27 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_2 = torch.ops.prims.prepare_softmax_online.default(add_76, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_54: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_2[0]
        getitem_55: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_2[1];  prepare_softmax_online_default_2 = None
        sub_tensor_2: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_76, getitem_54);  add_76 = getitem_54 = None
        exp_default_2: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        div_28: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_2, getitem_55);  exp_default_2 = getitem_55 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_372: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.float16)
        expand_39: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_372, [4, 12, 476, 476]);  convert_element_type_372 = None
        view_210: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_39, [48, 476, 476]);  expand_39 = None
        expand_40: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_104, [4, 12, 476, 64]);  permute_104 = None
        clone_67: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_211: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [48, 476, 64]);  clone_67 = None
        bmm_19: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [4, 12, 476, 64]);  bmm_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_106: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_68: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_213: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [4, 476, 768]);  clone_68 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_375: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.float16);  primals_160 = None
        convert_element_type_376: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.float16);  primals_159 = None
        view_214: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [1904, 768]);  view_213 = None
        permute_107: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_376, [1, 0]);  convert_element_type_376 = None
        addmm_57: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_375, view_214, permute_107);  convert_element_type_375 = None
        view_215: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [4, 476, 768]);  addmm_57 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_77: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_215, add_75);  view_215 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_78: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
        mul_57: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
        mul_58: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, primals_161)
        add_79: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, primals_162);  mul_58 = primals_162 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_380: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.float16);  primals_164 = None
        convert_element_type_381: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.float16);  primals_163 = None
        convert_element_type_382: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.float16)
        view_216: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_382, [1904, 768]);  convert_element_type_382 = None
        permute_108: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_381, [1, 0]);  convert_element_type_381 = None
        addmm_58: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_380, view_216, permute_108);  convert_element_type_380 = None
        view_217: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_59: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        div_29: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_217, 1.4142135623730951);  view_217 = None
        erf_9: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_29);  div_29 = None
        add_80: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1.0);  erf_9 = None
        mul_60: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_80);  mul_59 = add_80 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_386: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.float16);  primals_166 = None
        convert_element_type_387: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.float16);  primals_165 = None
        view_218: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [1904, 3072]);  mul_60 = None
        permute_109: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_387, [1, 0]);  convert_element_type_387 = None
        addmm_59: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_386, view_218, permute_109);  convert_element_type_386 = None
        view_219: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [4, 476, 768]);  addmm_59 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_219, add_79);  view_219 = add_79 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_61: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
        mul_62: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, primals_167)
        add_83: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, primals_168);  mul_62 = primals_168 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_391: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.float16);  primals_170 = None
        convert_element_type_392: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.float16);  primals_169 = None
        convert_element_type_393: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.float16)
        view_220: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_393, [1904, 768]);  convert_element_type_393 = None
        permute_110: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_392, [1, 0]);  convert_element_type_392 = None
        addmm_60: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_391, view_220, permute_110);  convert_element_type_391 = None
        view_221: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [4, 476, 768]);  addmm_60 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_397: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.float16);  primals_172 = None
        convert_element_type_398: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.float16);  primals_171 = None
        permute_111: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_398, [1, 0]);  convert_element_type_398 = None
        addmm_61: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_397, view_220, permute_111);  convert_element_type_397 = None
        view_223: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [4, 476, 768]);  addmm_61 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_403: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.float16);  primals_174 = None
        convert_element_type_404: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.float16);  primals_173 = None
        permute_112: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_404, [1, 0]);  convert_element_type_404 = None
        addmm_62: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_403, view_220, permute_112);  convert_element_type_403 = None
        view_225: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [4, 476, 768]);  addmm_62 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_226: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [4, 476, 12, 64]);  view_221 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_113: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_227: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [4, 476, 12, 64]);  view_223 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_114: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_228: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [4, 476, 12, 64]);  view_225 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_115: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_116: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        expand_41: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_113, [4, 12, 476, 64]);  permute_113 = None
        clone_71: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_229: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [48, 476, 64]);  clone_71 = None
        expand_42: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_116, [4, 12, 64, 476]);  permute_116 = None
        clone_72: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_230: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [48, 64, 476]);  clone_72 = None
        bmm_20: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [4, 12, 476, 476]);  bmm_20 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_30: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_84: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_30, mul);  div_30 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_1 = torch.ops.prims.prepare_softmax_online.default(add_84, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_52: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_1[0]
        getitem_53: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default_1[1];  prepare_softmax_online_default_1 = None
        sub_tensor_1: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_84, getitem_52);  add_84 = getitem_52 = None
        exp_default_1: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        div_31: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default_1, getitem_53);  exp_default_1 = getitem_53 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_411: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.float16)
        expand_43: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_411, [4, 12, 476, 476]);  convert_element_type_411 = None
        view_232: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_43, [48, 476, 476]);  expand_43 = None
        expand_44: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_115, [4, 12, 476, 64]);  permute_115 = None
        clone_74: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_233: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [48, 476, 64]);  clone_74 = None
        bmm_21: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_232, view_233)
        view_234: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [4, 12, 476, 64]);  bmm_21 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_117: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_75: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_235: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [4, 476, 768]);  clone_75 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_414: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.float16);  primals_176 = None
        convert_element_type_415: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.float16);  primals_175 = None
        view_236: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [1904, 768]);  view_235 = None
        permute_118: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_415, [1, 0]);  convert_element_type_415 = None
        addmm_63: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_414, view_236, permute_118);  convert_element_type_414 = None
        view_237: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [4, 476, 768]);  addmm_63 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_237, add_83);  view_237 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_86: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
        mul_63: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
        mul_64: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, primals_177)
        add_87: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, primals_178);  mul_64 = primals_178 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_419: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.float16);  primals_180 = None
        convert_element_type_420: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.float16);  primals_179 = None
        convert_element_type_421: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.float16)
        view_238: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [1904, 768]);  convert_element_type_421 = None
        permute_119: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_420, [1, 0]);  convert_element_type_420 = None
        addmm_64: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_419, view_238, permute_119);  convert_element_type_419 = None
        view_239: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_65: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        div_32: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_239, 1.4142135623730951);  view_239 = None
        erf_10: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_32);  div_32 = None
        add_88: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1.0);  erf_10 = None
        mul_66: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, add_88);  mul_65 = add_88 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_425: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.float16);  primals_182 = None
        convert_element_type_426: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.float16);  primals_181 = None
        view_240: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_66, [1904, 3072]);  mul_66 = None
        permute_120: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_426, [1, 0]);  convert_element_type_426 = None
        addmm_65: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_425, view_240, permute_120);  convert_element_type_425 = None
        view_241: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [4, 476, 768]);  addmm_65 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_89: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_241, add_87);  view_241 = add_87 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_67: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
        mul_68: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, primals_183)
        add_91: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_68, primals_184);  mul_68 = primals_184 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_430: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.float16);  primals_186 = None
        convert_element_type_431: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.float16);  primals_185 = None
        convert_element_type_432: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float16)
        view_242: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_432, [1904, 768]);  convert_element_type_432 = None
        permute_121: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_431, [1, 0]);  convert_element_type_431 = None
        addmm_66: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_430, view_242, permute_121);  convert_element_type_430 = None
        view_243: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [4, 476, 768]);  addmm_66 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_436: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.float16);  primals_188 = None
        convert_element_type_437: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.float16);  primals_187 = None
        permute_122: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_437, [1, 0]);  convert_element_type_437 = None
        addmm_67: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_436, view_242, permute_122);  convert_element_type_436 = None
        view_245: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [4, 476, 768]);  addmm_67 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_442: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.float16);  primals_190 = None
        convert_element_type_443: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.float16);  primals_189 = None
        permute_123: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_443, [1, 0]);  convert_element_type_443 = None
        addmm_68: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_442, view_242, permute_123);  convert_element_type_442 = None
        view_247: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [4, 476, 768]);  addmm_68 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_248: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [4, 476, 12, 64]);  view_243 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_124: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_249: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [4, 476, 12, 64]);  view_245 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_125: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_250: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [4, 476, 12, 64]);  view_247 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_126: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_127: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        expand_45: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_124, [4, 12, 476, 64]);  permute_124 = None
        clone_78: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_251: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [48, 476, 64]);  clone_78 = None
        expand_46: "f16[4, 12, 64, 476][365568, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_127, [4, 12, 64, 476]);  permute_127 = None
        clone_79: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_252: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [48, 64, 476]);  clone_79 = None
        bmm_22: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_251, view_252)
        view_253: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [4, 12, 476, 476]);  bmm_22 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_33: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_92: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div_33, mul);  div_33 = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(add_92, -1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        getitem_50: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default[0]
        getitem_51: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, getitem_50);  add_92 = getitem_50 = None
        exp_default: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_34: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_default, getitem_51);  exp_default = getitem_51 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        convert_element_type_450: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.float16)
        expand_47: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_450, [4, 12, 476, 476]);  convert_element_type_450 = None
        view_254: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(expand_47, [48, 476, 476]);  expand_47 = None
        expand_48: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_126, [4, 12, 476, 64]);  permute_126 = None
        clone_81: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_255: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [48, 476, 64]);  clone_81 = None
        bmm_23: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [4, 12, 476, 64]);  bmm_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_128: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_82: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_257: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [4, 476, 768]);  clone_82 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_453: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.float16);  primals_192 = None
        convert_element_type_454: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.float16);  primals_191 = None
        view_258: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [1904, 768]);  view_257 = None
        permute_129: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_454, [1, 0]);  convert_element_type_454 = None
        addmm_69: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_453, view_258, permute_129);  convert_element_type_453 = None
        view_259: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [4, 476, 768]);  addmm_69 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_93: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_259, add_91);  view_259 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_94: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
        mul_69: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
        mul_70: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, primals_193)
        add_95: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_70, primals_194);  mul_70 = primals_194 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_458: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.float16);  primals_196 = None
        convert_element_type_459: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.float16);  primals_195 = None
        convert_element_type_460: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.float16)
        view_260: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_460, [1904, 768]);  convert_element_type_460 = None
        permute_130: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_459, [1, 0]);  convert_element_type_459 = None
        addmm_70: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_458, view_260, permute_130);  convert_element_type_458 = None
        view_261: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [4, 476, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_71: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        div_35: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_261, 1.4142135623730951);  view_261 = None
        erf_11: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_35);  div_35 = None
        add_96: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1.0);  erf_11 = None
        mul_72: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, add_96);  mul_71 = add_96 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_464: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.float16);  primals_198 = None
        convert_element_type_465: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.float16);  primals_197 = None
        view_262: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_72, [1904, 3072]);  mul_72 = None
        permute_131: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_465, [1, 0]);  convert_element_type_465 = None
        addmm_71: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_464, view_262, permute_131);  convert_element_type_464 = None
        view_263: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [4, 476, 768]);  addmm_71 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_97: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_263, add_95);  view_263 = add_95 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[4, 476, 1][476, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_73: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
        mul_74: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, primals_199)
        add_99: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, primals_200);  mul_74 = primals_200 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:403 in forward, code: first_token_tensor = hidden_states[:, 0]
        select: "f32[4, 768][365568, 1]cuda:0" = torch.ops.aten.select.int(add_99, 1, 0)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:404 in forward, code: pooled_output = self.dense(first_token_tensor)
        convert_element_type_469: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_202, torch.float16);  primals_202 = None
        convert_element_type_470: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.float16);  primals_201 = None
        convert_element_type_471: "f16[4, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(select, torch.float16);  select = None
        permute_132: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_470, [1, 0]);  convert_element_type_470 = None
        addmm_72: "f16[4, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_469, convert_element_type_471, permute_132);  convert_element_type_469 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:405 in forward, code: pooled_output = self.activation(pooled_output)
        tanh: "f16[4, 768][768, 1]cuda:0" = torch.ops.aten.tanh.default(addmm_72);  addmm_72 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:404 in forward, code: pooled_output = self.dense(first_token_tensor)
        permute_133: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_137: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_141: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_38: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_145: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_150: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_151: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_152: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_153: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_158: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_162: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_166: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_40: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_170: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_174: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_42: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_178: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_183: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_184: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_185: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_186: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_191: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_195: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_199: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_44: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_203: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_207: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_46: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_211: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_216: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_217: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_218: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_219: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_224: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_228: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_232: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_48: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_236: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_240: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_50: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_244: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_249: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_250: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_251: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_252: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_257: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_261: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_265: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_52: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_269: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_273: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_54: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_277: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_282: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_283: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_284: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_285: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_290: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_294: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_298: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_56: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_302: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_306: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_58: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_310: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_315: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_316: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_317: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_318: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_323: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_327: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_331: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_60: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_335: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_339: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_62: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_343: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_348: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_349: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_350: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_351: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_356: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_360: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_364: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_64: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_368: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_372: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_66: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_376: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_381: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_382: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_383: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_384: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_389: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_393: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_397: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_68: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_401: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_405: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_70: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_409: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_414: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_415: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_416: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_417: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_422: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_426: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_430: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_72: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_434: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_438: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_74: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_442: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_447: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_448: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_449: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_450: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_455: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_459: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_463: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_76: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_467: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_471: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_78: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_475: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_480: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_481: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_482: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_483: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_488: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_492: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_496: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_80: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        permute_500: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        permute_504: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_82: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        permute_508: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_513: "f16[48, 476, 476][226576, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_514: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_515: "f16[48, 64, 476][30464, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_516: "f16[48, 476, 64][30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_521: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_525: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_529: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:243 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_84: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (add_3, add_11, add_19, add_27, add_35, add_43, add_51, add_59, add_67, add_75, add_83, add_91, add_99, tanh, primals_2, primals_4, primals_7, primals_17, primals_23, primals_33, primals_39, primals_49, primals_55, primals_65, primals_71, primals_81, primals_87, primals_97, primals_103, primals_113, primals_119, primals_129, primals_135, primals_145, primals_151, primals_161, primals_167, primals_177, primals_183, primals_193, primals_199, mul, unsqueeze_2, mul_1, view, bmm, getitem_72, getitem_73, view_16, mul_3, view_18, addmm_4, view_20, mul_7, view_22, div_4, view_38, mul_9, view_40, addmm_10, view_42, mul_13, view_44, div_7, view_60, mul_15, view_62, addmm_16, view_64, mul_19, view_66, div_10, view_82, mul_21, view_84, addmm_22, view_86, mul_25, view_88, div_13, view_104, mul_27, view_106, addmm_28, view_108, mul_31, view_110, div_16, view_126, mul_33, view_128, addmm_34, view_130, mul_37, view_132, div_19, view_148, mul_39, view_150, addmm_40, view_152, mul_43, view_154, div_22, view_170, mul_45, view_172, addmm_46, view_174, mul_49, view_176, div_25, view_192, mul_51, view_194, addmm_52, view_196, mul_55, view_198, div_28, view_214, mul_57, view_216, addmm_58, view_218, mul_61, view_220, div_31, view_236, mul_63, view_238, addmm_64, view_240, mul_67, view_242, div_34, view_258, mul_69, view_260, addmm_70, view_262, mul_73, convert_element_type_471, tanh, permute_133, div_36, permute_137, permute_141, div_38, permute_145, permute_150, permute_151, permute_152, permute_153, permute_158, permute_162, permute_166, div_40, permute_170, permute_174, div_42, permute_178, permute_183, permute_184, permute_185, permute_186, permute_191, permute_195, permute_199, div_44, permute_203, permute_207, div_46, permute_211, permute_216, permute_217, permute_218, permute_219, permute_224, permute_228, permute_232, div_48, permute_236, permute_240, div_50, permute_244, permute_249, permute_250, permute_251, permute_252, permute_257, permute_261, permute_265, div_52, permute_269, permute_273, div_54, permute_277, permute_282, permute_283, permute_284, permute_285, permute_290, permute_294, permute_298, div_56, permute_302, permute_306, div_58, permute_310, permute_315, permute_316, permute_317, permute_318, permute_323, permute_327, permute_331, div_60, permute_335, permute_339, div_62, permute_343, permute_348, permute_349, permute_350, permute_351, permute_356, permute_360, permute_364, div_64, permute_368, permute_372, div_66, permute_376, permute_381, permute_382, permute_383, permute_384, permute_389, permute_393, permute_397, div_68, permute_401, permute_405, div_70, permute_409, permute_414, permute_415, permute_416, permute_417, permute_422, permute_426, permute_430, div_72, permute_434, permute_438, div_74, permute_442, permute_447, permute_448, permute_449, permute_450, permute_455, permute_459, permute_463, div_76, permute_467, permute_471, div_78, permute_475, permute_480, permute_481, permute_482, permute_483, permute_488, permute_492, permute_496, div_80, permute_500, permute_504, div_82, permute_508, permute_513, permute_514, permute_515, permute_516, permute_521, permute_525, permute_529, div_84)
