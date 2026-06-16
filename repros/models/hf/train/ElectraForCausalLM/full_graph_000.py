class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[64, 512][512, 1]cuda:0", primals_2: "i64[64, 512][512, 1]cuda:0", primals_3: "i64[1, 512][512, 1]cuda:0", primals_4: "i64[1, 512][512, 1]cuda:0", primals_5: "f32[30522, 128][128, 1]cuda:0", primals_6: "f32[2, 128][128, 1]cuda:0", primals_7: "f32[512, 128][128, 1]cuda:0", primals_8: "f32[128][1]cuda:0", primals_9: "f32[128][1]cuda:0", primals_10: "f32[256, 128][128, 1]cuda:0", primals_11: "f32[256][1]cuda:0", primals_12: "f32[256, 256][256, 1]cuda:0", primals_13: "f32[256][1]cuda:0", primals_14: "f32[256, 256][256, 1]cuda:0", primals_15: "f32[256][1]cuda:0", primals_16: "f32[256, 256][256, 1]cuda:0", primals_17: "f32[256][1]cuda:0", primals_18: "f32[256, 256][256, 1]cuda:0", primals_19: "f32[256][1]cuda:0", primals_20: "f32[256][1]cuda:0", primals_21: "f32[256][1]cuda:0", primals_22: "f32[1024, 256][256, 1]cuda:0", primals_23: "f32[1024][1]cuda:0", primals_24: "f32[256, 1024][1024, 1]cuda:0", primals_25: "f32[256][1]cuda:0", primals_26: "f32[256][1]cuda:0", primals_27: "f32[256][1]cuda:0", primals_28: "f32[256, 256][256, 1]cuda:0", primals_29: "f32[256][1]cuda:0", primals_30: "f32[256, 256][256, 1]cuda:0", primals_31: "f32[256][1]cuda:0", primals_32: "f32[256, 256][256, 1]cuda:0", primals_33: "f32[256][1]cuda:0", primals_34: "f32[256, 256][256, 1]cuda:0", primals_35: "f32[256][1]cuda:0", primals_36: "f32[256][1]cuda:0", primals_37: "f32[256][1]cuda:0", primals_38: "f32[1024, 256][256, 1]cuda:0", primals_39: "f32[1024][1]cuda:0", primals_40: "f32[256, 1024][1024, 1]cuda:0", primals_41: "f32[256][1]cuda:0", primals_42: "f32[256][1]cuda:0", primals_43: "f32[256][1]cuda:0", primals_44: "f32[256, 256][256, 1]cuda:0", primals_45: "f32[256][1]cuda:0", primals_46: "f32[256, 256][256, 1]cuda:0", primals_47: "f32[256][1]cuda:0", primals_48: "f32[256, 256][256, 1]cuda:0", primals_49: "f32[256][1]cuda:0", primals_50: "f32[256, 256][256, 1]cuda:0", primals_51: "f32[256][1]cuda:0", primals_52: "f32[256][1]cuda:0", primals_53: "f32[256][1]cuda:0", primals_54: "f32[1024, 256][256, 1]cuda:0", primals_55: "f32[1024][1]cuda:0", primals_56: "f32[256, 1024][1024, 1]cuda:0", primals_57: "f32[256][1]cuda:0", primals_58: "f32[256][1]cuda:0", primals_59: "f32[256][1]cuda:0", primals_60: "f32[256, 256][256, 1]cuda:0", primals_61: "f32[256][1]cuda:0", primals_62: "f32[256, 256][256, 1]cuda:0", primals_63: "f32[256][1]cuda:0", primals_64: "f32[256, 256][256, 1]cuda:0", primals_65: "f32[256][1]cuda:0", primals_66: "f32[256, 256][256, 1]cuda:0", primals_67: "f32[256][1]cuda:0", primals_68: "f32[256][1]cuda:0", primals_69: "f32[256][1]cuda:0", primals_70: "f32[1024, 256][256, 1]cuda:0", primals_71: "f32[1024][1]cuda:0", primals_72: "f32[256, 1024][1024, 1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_74: "f32[256][1]cuda:0", primals_75: "f32[256][1]cuda:0", primals_76: "f32[256, 256][256, 1]cuda:0", primals_77: "f32[256][1]cuda:0", primals_78: "f32[256, 256][256, 1]cuda:0", primals_79: "f32[256][1]cuda:0", primals_80: "f32[256, 256][256, 1]cuda:0", primals_81: "f32[256][1]cuda:0", primals_82: "f32[256, 256][256, 1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_86: "f32[1024, 256][256, 1]cuda:0", primals_87: "f32[1024][1]cuda:0", primals_88: "f32[256, 1024][1024, 1]cuda:0", primals_89: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_91: "f32[256][1]cuda:0", primals_92: "f32[256, 256][256, 1]cuda:0", primals_93: "f32[256][1]cuda:0", primals_94: "f32[256, 256][256, 1]cuda:0", primals_95: "f32[256][1]cuda:0", primals_96: "f32[256, 256][256, 1]cuda:0", primals_97: "f32[256][1]cuda:0", primals_98: "f32[256, 256][256, 1]cuda:0", primals_99: "f32[256][1]cuda:0", primals_100: "f32[256][1]cuda:0", primals_101: "f32[256][1]cuda:0", primals_102: "f32[1024, 256][256, 1]cuda:0", primals_103: "f32[1024][1]cuda:0", primals_104: "f32[256, 1024][1024, 1]cuda:0", primals_105: "f32[256][1]cuda:0", primals_106: "f32[256][1]cuda:0", primals_107: "f32[256][1]cuda:0", primals_108: "f32[256, 256][256, 1]cuda:0", primals_109: "f32[256][1]cuda:0", primals_110: "f32[256, 256][256, 1]cuda:0", primals_111: "f32[256][1]cuda:0", primals_112: "f32[256, 256][256, 1]cuda:0", primals_113: "f32[256][1]cuda:0", primals_114: "f32[256, 256][256, 1]cuda:0", primals_115: "f32[256][1]cuda:0", primals_116: "f32[256][1]cuda:0", primals_117: "f32[256][1]cuda:0", primals_118: "f32[1024, 256][256, 1]cuda:0", primals_119: "f32[1024][1]cuda:0", primals_120: "f32[256, 1024][1024, 1]cuda:0", primals_121: "f32[256][1]cuda:0", primals_122: "f32[256][1]cuda:0", primals_123: "f32[256][1]cuda:0", primals_124: "f32[256, 256][256, 1]cuda:0", primals_125: "f32[256][1]cuda:0", primals_126: "f32[256, 256][256, 1]cuda:0", primals_127: "f32[256][1]cuda:0", primals_128: "f32[256, 256][256, 1]cuda:0", primals_129: "f32[256][1]cuda:0", primals_130: "f32[256, 256][256, 1]cuda:0", primals_131: "f32[256][1]cuda:0", primals_132: "f32[256][1]cuda:0", primals_133: "f32[256][1]cuda:0", primals_134: "f32[1024, 256][256, 1]cuda:0", primals_135: "f32[1024][1]cuda:0", primals_136: "f32[256, 1024][1024, 1]cuda:0", primals_137: "f32[256][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_139: "f32[256][1]cuda:0", primals_140: "f32[256, 256][256, 1]cuda:0", primals_141: "f32[256][1]cuda:0", primals_142: "f32[256, 256][256, 1]cuda:0", primals_143: "f32[256][1]cuda:0", primals_144: "f32[256, 256][256, 1]cuda:0", primals_145: "f32[256][1]cuda:0", primals_146: "f32[256, 256][256, 1]cuda:0", primals_147: "f32[256][1]cuda:0", primals_148: "f32[256][1]cuda:0", primals_149: "f32[256][1]cuda:0", primals_150: "f32[1024, 256][256, 1]cuda:0", primals_151: "f32[1024][1]cuda:0", primals_152: "f32[256, 1024][1024, 1]cuda:0", primals_153: "f32[256][1]cuda:0", primals_154: "f32[256][1]cuda:0", primals_155: "f32[256][1]cuda:0", primals_156: "f32[256, 256][256, 1]cuda:0", primals_157: "f32[256][1]cuda:0", primals_158: "f32[256, 256][256, 1]cuda:0", primals_159: "f32[256][1]cuda:0", primals_160: "f32[256, 256][256, 1]cuda:0", primals_161: "f32[256][1]cuda:0", primals_162: "f32[256, 256][256, 1]cuda:0", primals_163: "f32[256][1]cuda:0", primals_164: "f32[256][1]cuda:0", primals_165: "f32[256][1]cuda:0", primals_166: "f32[1024, 256][256, 1]cuda:0", primals_167: "f32[1024][1]cuda:0", primals_168: "f32[256, 1024][1024, 1]cuda:0", primals_169: "f32[256][1]cuda:0", primals_170: "f32[256][1]cuda:0", primals_171: "f32[256][1]cuda:0", primals_172: "f32[256, 256][256, 1]cuda:0", primals_173: "f32[256][1]cuda:0", primals_174: "f32[256, 256][256, 1]cuda:0", primals_175: "f32[256][1]cuda:0", primals_176: "f32[256, 256][256, 1]cuda:0", primals_177: "f32[256][1]cuda:0", primals_178: "f32[256, 256][256, 1]cuda:0", primals_179: "f32[256][1]cuda:0", primals_180: "f32[256][1]cuda:0", primals_181: "f32[256][1]cuda:0", primals_182: "f32[1024, 256][256, 1]cuda:0", primals_183: "f32[1024][1]cuda:0", primals_184: "f32[256, 1024][1024, 1]cuda:0", primals_185: "f32[256][1]cuda:0", primals_186: "f32[256][1]cuda:0", primals_187: "f32[256][1]cuda:0", primals_188: "f32[256, 256][256, 1]cuda:0", primals_189: "f32[256][1]cuda:0", primals_190: "f32[256, 256][256, 1]cuda:0", primals_191: "f32[256][1]cuda:0", primals_192: "f32[256, 256][256, 1]cuda:0", primals_193: "f32[256][1]cuda:0", primals_194: "f32[256, 256][256, 1]cuda:0", primals_195: "f32[256][1]cuda:0", primals_196: "f32[256][1]cuda:0", primals_197: "f32[256][1]cuda:0", primals_198: "f32[1024, 256][256, 1]cuda:0", primals_199: "f32[1024][1]cuda:0", primals_200: "f32[256, 1024][1024, 1]cuda:0", primals_201: "f32[256][1]cuda:0", primals_202: "f32[256][1]cuda:0", primals_203: "f32[256][1]cuda:0", primals_204: "f32[128, 256][256, 1]cuda:0", primals_205: "f32[128][1]cuda:0", primals_206: "f32[128][1]cuda:0", primals_207: "f32[128][1]cuda:0", primals_208: "f32[30522][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:101 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.expand.default(primals_4, [1, -1]);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:102 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.gather.default(expand, 1, primals_3);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[64, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(gather, [64, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.embedding.default(primals_5, primals_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.embedding.default(primals_6, expand_1);  primals_6 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:110 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.embedding.default(primals_7, primals_3);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_2: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_8)
        add_3: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_9);  mul_1 = primals_9 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[37][1]cuda:0" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:116 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_36: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_2: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_3: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        convert_element_type: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_1: "bf16[256, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_2: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[32768, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [32768, 128]);  convert_element_type_2 = None
        permute: "bf16[128, 256][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [64, 512, 256]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_6: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_7: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        view_2: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32768, 256])
        permute_1: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view_2, permute_1);  convert_element_type_6 = None
        view_3: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [64, 512, 256]);  addmm_1 = None
        view_4: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [64, 512, -1, 64]);  view_3 = None
        permute_2: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_11: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_12: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        permute_3: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_2: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_11, view_2, permute_3);  convert_element_type_11 = None
        view_6: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [64, 512, 256]);  addmm_2 = None
        view_7: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [64, 512, -1, 64]);  view_6 = None
        permute_4: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_16: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_17: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        permute_5: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        addmm_3: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_16, view_2, permute_5);  convert_element_type_16 = None
        view_9: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [64, 512, 256]);  addmm_3 = None
        view_10: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [64, 512, -1, 64]);  view_9 = None
        permute_6: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_4: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_2, 0.3535533905932738);  permute_2 = None
        permute_7: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_5: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_7, 0.3535533905932738);  permute_7 = None
        expand_3: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_4, [64, 4, 512, 64]);  mul_4 = None
        clone: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_11: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [256, 512, 64]);  clone = None
        expand_4: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_5, [64, 4, 64, 512]);  mul_5 = None
        clone_1: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_12: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [256, 64, 512]);  clone_1 = None
        bmm: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_11, view_12)
        view_13: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [64, 4, 512, 512])
        convert_element_type_23: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32)
        amax: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_23, [-1], True)
        sub_1: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, amax);  convert_element_type_23 = None
        exp: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_24: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_13, -inf);  view_13 = None
        logical_not: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_1: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_24);  convert_element_type_24 = None
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_35: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_35: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_35, torch.bfloat16);  inductor_random_default_35 = None
        gt_1: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_35, 0.1);  convert_element_type_default_35 = None
        mul_6: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, where_1);  where_1 = None
        mul_7: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_5: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_7, [64, 4, 512, 512]);  mul_7 = None
        view_14: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_5, [256, 512, 512]);  expand_5 = None
        expand_6: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_6, [64, 4, 512, 64]);  permute_6 = None
        clone_2: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_15: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [256, 512, 64]);  clone_2 = None
        bmm_1: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_14, view_15)
        view_16: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [64, 4, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_3: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_17: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [64, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_27: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_28: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        view_18: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [32768, 256]);  view_17 = None
        permute_9: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        addmm_4: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_27, view_18, permute_9);  convert_element_type_27 = None
        view_19: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [64, 512, 256]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_34: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_34: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_34, torch.bfloat16);  inductor_random_default_34 = None
        gt_2: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_34, 0.1);  convert_element_type_default_34 = None
        mul_8: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_19);  view_19 = None
        mul_9: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, view_1);  mul_9 = view_1 = None
        convert_element_type_32: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True)
        getitem_2: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_8: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_3);  convert_element_type_32 = None
        mul_10: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_20);  mul_10 = None
        add_9: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_21);  mul_11 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_33: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_34: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_35: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16)
        view_20: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [32768, 256]);  convert_element_type_35 = None
        permute_10: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_5: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_33, view_20, permute_10);  convert_element_type_33 = None
        view_21: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_39: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_12: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.5)
        mul_13: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.7071067811865476);  convert_element_type_39 = None
        erf: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_10);  mul_12 = add_10 = None
        convert_element_type_40: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_41: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_42: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        view_22: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_40, [32768, 1024]);  convert_element_type_40 = None
        permute_11: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        addmm_6: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_41, view_22, permute_11);  convert_element_type_41 = None
        view_23: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [64, 512, 256]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_33: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_33: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_3: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_33, 0.1);  convert_element_type_default_33 = None
        mul_15: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_23);  view_23 = None
        mul_16: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, add_9);  mul_16 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        mul_17: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_18: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, primals_26)
        add_13: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, primals_27);  mul_18 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_46: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_47: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_48: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16)
        view_24: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_48, [32768, 256]);  convert_element_type_48 = None
        permute_12: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        addmm_7: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_46, view_24, permute_12);  convert_element_type_46 = None
        view_25: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [64, 512, 256]);  addmm_7 = None
        view_26: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [64, 512, -1, 64]);  view_25 = None
        permute_13: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_52: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_53: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_14: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        addmm_8: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_52, view_24, permute_14);  convert_element_type_52 = None
        view_28: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [64, 512, 256]);  addmm_8 = None
        view_29: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [64, 512, -1, 64]);  view_28 = None
        permute_15: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_58: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_59: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_16: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        addmm_9: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_58, view_24, permute_16);  convert_element_type_58 = None
        view_31: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [64, 512, 256]);  addmm_9 = None
        view_32: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [64, 512, -1, 64]);  view_31 = None
        permute_17: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_19: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_13, 0.3535533905932738);  permute_13 = None
        permute_18: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_20: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_18, 0.3535533905932738);  permute_18 = None
        expand_7: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_19, [64, 4, 512, 64]);  mul_19 = None
        clone_4: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_33: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [256, 512, 64]);  clone_4 = None
        expand_8: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_20, [64, 4, 64, 512]);  mul_20 = None
        clone_5: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_34: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [256, 64, 512]);  clone_5 = None
        bmm_2: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_33, view_34)
        view_35: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [64, 4, 512, 512]);  bmm_2 = None
        convert_element_type_66: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32)
        amax_1: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_66, [-1], True)
        sub_4: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_66, amax_1);  convert_element_type_66 = amax_1 = None
        exp_1: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_67: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        eq_1: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_35, -inf);  view_35 = None
        logical_not_2: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_1, convert_element_type_67);  logical_not_3 = convert_element_type_67 = None
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_32: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_32: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_32, torch.bfloat16);  inductor_random_default_32 = None
        gt_4: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_32, 0.1);  convert_element_type_default_32 = None
        mul_21: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, where_3)
        mul_22: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, 1.1111111111111112);  mul_21 = None
        expand_9: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_22, [64, 4, 512, 512]);  mul_22 = None
        view_36: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_9, [256, 512, 512]);  expand_9 = None
        expand_10: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_17, [64, 4, 512, 64]);  permute_17 = None
        clone_6: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_37: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [256, 512, 64]);  clone_6 = None
        bmm_3: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_36, view_37)
        view_38: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [64, 4, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_7: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_39: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [64, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_70: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_71: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        view_40: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [32768, 256]);  view_39 = None
        permute_20: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_10: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_70, view_40, permute_20);  convert_element_type_70 = None
        view_41: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [64, 512, 256]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_31: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_31: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_5: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_31, 0.1);  convert_element_type_default_31 = None
        mul_23: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_41);  view_41 = None
        mul_24: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, add_13);  mul_24 = add_13 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        mul_25: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_26: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, primals_36)
        add_17: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, primals_37);  mul_26 = primals_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_75: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_76: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_77: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16)
        view_42: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_77, [32768, 256]);  convert_element_type_77 = None
        permute_21: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        addmm_11: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_75, view_42, permute_21);  convert_element_type_75 = None
        view_43: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_81: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_27: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.5)
        mul_28: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.7071067811865476);  convert_element_type_81 = None
        erf_1: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_29: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, add_18);  mul_27 = add_18 = None
        convert_element_type_82: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_83: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_84: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        view_44: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_82, [32768, 1024]);  convert_element_type_82 = None
        permute_22: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        addmm_12: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_83, view_44, permute_22);  convert_element_type_83 = None
        view_45: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [64, 512, 256]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_30: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_30: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_30, torch.bfloat16);  inductor_random_default_30 = None
        gt_6: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_30, 0.1);  convert_element_type_default_30 = None
        mul_30: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, view_45);  view_45 = None
        mul_31: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, add_17);  mul_31 = add_17 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_32: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_33: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, primals_42)
        add_21: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, primals_43);  mul_33 = primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_88: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_89: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_90: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16)
        view_46: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_90, [32768, 256]);  convert_element_type_90 = None
        permute_23: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_89, [1, 0]);  convert_element_type_89 = None
        addmm_13: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_88, view_46, permute_23);  convert_element_type_88 = None
        view_47: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [64, 512, 256]);  addmm_13 = None
        view_48: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_47, [64, 512, -1, 64]);  view_47 = None
        permute_24: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_94: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convert_element_type_95: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        permute_25: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        addmm_14: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_94, view_46, permute_25);  convert_element_type_94 = None
        view_50: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [64, 512, 256]);  addmm_14 = None
        view_51: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [64, 512, -1, 64]);  view_50 = None
        permute_26: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_100: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_101: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        permute_27: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        addmm_15: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_100, view_46, permute_27);  convert_element_type_100 = None
        view_53: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [64, 512, 256]);  addmm_15 = None
        view_54: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [64, 512, -1, 64]);  view_53 = None
        permute_28: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_34: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_24, 0.3535533905932738);  permute_24 = None
        permute_29: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_35: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_29, 0.3535533905932738);  permute_29 = None
        expand_11: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_34, [64, 4, 512, 64]);  mul_34 = None
        clone_8: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_55: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [256, 512, 64]);  clone_8 = None
        expand_12: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_35, [64, 4, 64, 512]);  mul_35 = None
        clone_9: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_56: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [256, 64, 512]);  clone_9 = None
        bmm_4: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_55, view_56)
        view_57: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [64, 4, 512, 512]);  bmm_4 = None
        convert_element_type_108: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        amax_2: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_108, [-1], True)
        sub_7: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_108, amax_2);  convert_element_type_108 = amax_2 = None
        exp_2: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_109: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        eq_2: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_57, -inf);  view_57 = None
        logical_not_4: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_1, convert_element_type_109);  logical_not_5 = convert_element_type_109 = None
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_29: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_29: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_29, torch.bfloat16);  inductor_random_default_29 = None
        gt_7: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_29, 0.1);  convert_element_type_default_29 = None
        mul_36: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, where_5)
        mul_37: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        expand_13: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_37, [64, 4, 512, 512]);  mul_37 = None
        view_58: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_13, [256, 512, 512]);  expand_13 = None
        expand_14: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_28, [64, 4, 512, 64]);  permute_28 = None
        clone_10: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_59: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [256, 512, 64]);  clone_10 = None
        bmm_5: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_58, view_59)
        view_60: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [64, 4, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_11: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [64, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_112: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_113: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        view_62: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [32768, 256]);  view_61 = None
        permute_31: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        addmm_16: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_112, view_62, permute_31);  convert_element_type_112 = None
        view_63: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [64, 512, 256]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_28: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_28: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_28, torch.bfloat16);  inductor_random_default_28 = None
        gt_8: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_28, 0.1);  convert_element_type_default_28 = None
        mul_38: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, view_63);  view_63 = None
        mul_39: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, add_21);  mul_39 = add_21 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_10: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_24: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_8: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
        mul_40: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_41: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, primals_52)
        add_25: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, primals_53);  mul_41 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_117: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_118: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convert_element_type_119: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16)
        view_64: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [32768, 256]);  convert_element_type_119 = None
        permute_32: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        addmm_17: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_117, view_64, permute_32);  convert_element_type_117 = None
        view_65: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_123: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        mul_42: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.5)
        mul_43: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476);  convert_element_type_123 = None
        erf_2: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_44: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, add_26);  mul_42 = add_26 = None
        convert_element_type_124: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_125: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_126: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        view_66: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_124, [32768, 1024]);  convert_element_type_124 = None
        permute_33: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        addmm_18: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_125, view_66, permute_33);  convert_element_type_125 = None
        view_67: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [64, 512, 256]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_27: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_27: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_9: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_27, 0.1);  convert_element_type_default_27 = None
        mul_45: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_67);  view_67 = None
        mul_46: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, add_25);  mul_46 = add_25 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_12: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_9: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, getitem_13);  add_27 = getitem_13 = None
        mul_47: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_48: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, primals_58)
        add_29: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, primals_59);  mul_48 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_130: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_131: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_132: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16)
        view_68: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_132, [32768, 256]);  convert_element_type_132 = None
        permute_34: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        addmm_19: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_130, view_68, permute_34);  convert_element_type_130 = None
        view_69: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [64, 512, 256]);  addmm_19 = None
        view_70: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [64, 512, -1, 64]);  view_69 = None
        permute_35: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_136: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_137: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        permute_36: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_137, [1, 0]);  convert_element_type_137 = None
        addmm_20: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_136, view_68, permute_36);  convert_element_type_136 = None
        view_72: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [64, 512, 256]);  addmm_20 = None
        view_73: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [64, 512, -1, 64]);  view_72 = None
        permute_37: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_142: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_143: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        permute_38: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_143, [1, 0]);  convert_element_type_143 = None
        addmm_21: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_142, view_68, permute_38);  convert_element_type_142 = None
        view_75: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [64, 512, 256]);  addmm_21 = None
        view_76: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [64, 512, -1, 64]);  view_75 = None
        permute_39: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_49: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_35, 0.3535533905932738);  permute_35 = None
        permute_40: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        mul_50: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_40, 0.3535533905932738);  permute_40 = None
        expand_15: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_49, [64, 4, 512, 64]);  mul_49 = None
        clone_12: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_77: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [256, 512, 64]);  clone_12 = None
        expand_16: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_50, [64, 4, 64, 512]);  mul_50 = None
        clone_13: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_78: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [256, 64, 512]);  clone_13 = None
        bmm_6: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_77, view_78)
        view_79: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [64, 4, 512, 512]);  bmm_6 = None
        convert_element_type_150: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_79, torch.float32)
        amax_3: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_150, [-1], True)
        sub_10: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, amax_3);  convert_element_type_150 = amax_3 = None
        exp_3: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_151: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        eq_3: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_79, -inf);  view_79 = None
        logical_not_6: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_1, convert_element_type_151);  logical_not_7 = convert_element_type_151 = None
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_26: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_26: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_26, torch.bfloat16);  inductor_random_default_26 = None
        gt_10: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_26, 0.1);  convert_element_type_default_26 = None
        mul_51: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, where_7)
        mul_52: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        expand_17: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_52, [64, 4, 512, 512]);  mul_52 = None
        view_80: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_17, [256, 512, 512]);  expand_17 = None
        expand_18: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_39, [64, 4, 512, 64]);  permute_39 = None
        clone_14: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_81: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [256, 512, 64]);  clone_14 = None
        bmm_7: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_80, view_81)
        view_82: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [64, 4, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_15: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_83: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [64, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_154: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_155: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        view_84: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [32768, 256]);  view_83 = None
        permute_42: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        addmm_22: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_154, view_84, permute_42);  convert_element_type_154 = None
        view_85: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [64, 512, 256]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_25: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_25: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_25, torch.bfloat16);  inductor_random_default_25 = None
        gt_11: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_25, 0.1);  convert_element_type_default_25 = None
        mul_53: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_85);  view_85 = None
        mul_54: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, add_29);  mul_54 = add_29 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_14: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_32: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_11: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_31, getitem_15);  add_31 = getitem_15 = None
        mul_55: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_56: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, primals_68)
        add_33: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, primals_69);  mul_56 = primals_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_159: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convert_element_type_160: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_161: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16)
        view_86: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_161, [32768, 256]);  convert_element_type_161 = None
        permute_43: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        addmm_23: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_159, view_86, permute_43);  convert_element_type_159 = None
        view_87: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_165: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None
        mul_57: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.5)
        mul_58: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.7071067811865476);  convert_element_type_165 = None
        erf_3: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_59: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, add_34);  mul_57 = add_34 = None
        convert_element_type_166: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_167: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_168: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        view_88: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_166, [32768, 1024]);  convert_element_type_166 = None
        permute_44: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        addmm_24: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_167, view_88, permute_44);  convert_element_type_167 = None
        view_89: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [64, 512, 256]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_24: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_24: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_24, torch.bfloat16);  inductor_random_default_24 = None
        gt_12: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_24, 0.1);  convert_element_type_default_24 = None
        mul_60: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, view_89);  view_89 = None
        mul_61: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, add_33);  mul_61 = add_33 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_16: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_12: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_35, getitem_17);  add_35 = getitem_17 = None
        mul_62: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_63: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, primals_74)
        add_37: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, primals_75);  mul_63 = primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_172: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        convert_element_type_173: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        convert_element_type_174: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16)
        view_90: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_174, [32768, 256]);  convert_element_type_174 = None
        permute_45: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        addmm_25: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_172, view_90, permute_45);  convert_element_type_172 = None
        view_91: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [64, 512, 256]);  addmm_25 = None
        view_92: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [64, 512, -1, 64]);  view_91 = None
        permute_46: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_178: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_179: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        permute_47: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        addmm_26: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_178, view_90, permute_47);  convert_element_type_178 = None
        view_94: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [64, 512, 256]);  addmm_26 = None
        view_95: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [64, 512, -1, 64]);  view_94 = None
        permute_48: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_184: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_185: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        permute_49: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_27: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_184, view_90, permute_49);  convert_element_type_184 = None
        view_97: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [64, 512, 256]);  addmm_27 = None
        view_98: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_97, [64, 512, -1, 64]);  view_97 = None
        permute_50: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_64: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_46, 0.3535533905932738);  permute_46 = None
        permute_51: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        mul_65: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_51, 0.3535533905932738);  permute_51 = None
        expand_19: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_64, [64, 4, 512, 64]);  mul_64 = None
        clone_16: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_99: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [256, 512, 64]);  clone_16 = None
        expand_20: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_65, [64, 4, 64, 512]);  mul_65 = None
        clone_17: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_100: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [256, 64, 512]);  clone_17 = None
        bmm_8: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_99, view_100)
        view_101: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [64, 4, 512, 512]);  bmm_8 = None
        convert_element_type_192: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_101, torch.float32)
        amax_4: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_192, [-1], True)
        sub_13: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, amax_4);  convert_element_type_192 = amax_4 = None
        exp_4: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_193: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        eq_4: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_101, -inf);  view_101 = None
        logical_not_8: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_1, convert_element_type_193);  logical_not_9 = convert_element_type_193 = None
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_23: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_23: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_13: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_23, 0.1);  convert_element_type_default_23 = None
        mul_66: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, where_9)
        mul_67: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        expand_21: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_67, [64, 4, 512, 512]);  mul_67 = None
        view_102: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_21, [256, 512, 512]);  expand_21 = None
        expand_22: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_50, [64, 4, 512, 64]);  permute_50 = None
        clone_18: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_103: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [256, 512, 64]);  clone_18 = None
        bmm_9: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_102, view_103)
        view_104: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [64, 4, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_19: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [64, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_196: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_197: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        view_106: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [32768, 256]);  view_105 = None
        permute_53: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        addmm_28: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_196, view_106, permute_53);  convert_element_type_196 = None
        view_107: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [64, 512, 256]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_22: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_22: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_14: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_22, 0.1);  convert_element_type_default_22 = None
        mul_68: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, view_107);  view_107 = None
        mul_69: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, add_37);  mul_69 = add_37 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_18: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_40: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_14: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_39, getitem_19);  add_39 = getitem_19 = None
        mul_70: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_71: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, primals_84)
        add_41: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, primals_85);  mul_71 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_201: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        convert_element_type_202: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_203: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16)
        view_108: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_203, [32768, 256]);  convert_element_type_203 = None
        permute_54: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        addmm_29: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_201, view_108, permute_54);  convert_element_type_201 = None
        view_109: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_207: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_72: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, 0.5)
        mul_73: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, 0.7071067811865476);  convert_element_type_207 = None
        erf_4: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_74: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, add_42);  mul_72 = add_42 = None
        convert_element_type_208: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_74, torch.bfloat16);  mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_209: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_210: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        view_110: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [32768, 1024]);  convert_element_type_208 = None
        permute_55: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        addmm_30: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_209, view_110, permute_55);  convert_element_type_209 = None
        view_111: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [64, 512, 256]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_21: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_21: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_15: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_21, 0.1);  convert_element_type_default_21 = None
        mul_75: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_111);  view_111 = None
        mul_76: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, add_41);  mul_76 = add_41 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_20: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_15: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_43, getitem_21);  add_43 = getitem_21 = None
        mul_77: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_78: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, primals_90)
        add_45: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, primals_91);  mul_78 = primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_214: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_215: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_216: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16)
        view_112: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_216, [32768, 256]);  convert_element_type_216 = None
        permute_56: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_215, [1, 0]);  convert_element_type_215 = None
        addmm_31: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_214, view_112, permute_56);  convert_element_type_214 = None
        view_113: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [64, 512, 256]);  addmm_31 = None
        view_114: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [64, 512, -1, 64]);  view_113 = None
        permute_57: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_220: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_221: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        permute_58: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        addmm_32: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_220, view_112, permute_58);  convert_element_type_220 = None
        view_116: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [64, 512, 256]);  addmm_32 = None
        view_117: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [64, 512, -1, 64]);  view_116 = None
        permute_59: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_226: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        convert_element_type_227: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        permute_60: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_227, [1, 0]);  convert_element_type_227 = None
        addmm_33: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_226, view_112, permute_60);  convert_element_type_226 = None
        view_119: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [64, 512, 256]);  addmm_33 = None
        view_120: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_119, [64, 512, -1, 64]);  view_119 = None
        permute_61: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_79: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_57, 0.3535533905932738);  permute_57 = None
        permute_62: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        mul_80: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_62, 0.3535533905932738);  permute_62 = None
        expand_23: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_79, [64, 4, 512, 64]);  mul_79 = None
        clone_20: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_121: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [256, 512, 64]);  clone_20 = None
        expand_24: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_80, [64, 4, 64, 512]);  mul_80 = None
        clone_21: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_122: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [256, 64, 512]);  clone_21 = None
        bmm_10: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_121, view_122)
        view_123: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [64, 4, 512, 512]);  bmm_10 = None
        convert_element_type_234: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_123, torch.float32)
        amax_5: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_234, [-1], True)
        sub_16: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_234, amax_5);  convert_element_type_234 = amax_5 = None
        exp_5: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_235: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        eq_5: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_123, -inf);  view_123 = None
        logical_not_10: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_1, convert_element_type_235);  logical_not_11 = convert_element_type_235 = None
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_20: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_20: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_20, torch.bfloat16);  inductor_random_default_20 = None
        gt_16: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_20, 0.1);  convert_element_type_default_20 = None
        mul_81: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, where_11)
        mul_82: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, 1.1111111111111112);  mul_81 = None
        expand_25: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_82, [64, 4, 512, 512]);  mul_82 = None
        view_124: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_25, [256, 512, 512]);  expand_25 = None
        expand_26: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_61, [64, 4, 512, 64]);  permute_61 = None
        clone_22: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_125: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [256, 512, 64]);  clone_22 = None
        bmm_11: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_124, view_125)
        view_126: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [64, 4, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_23: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_127: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [64, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_238: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_239: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        view_128: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [32768, 256]);  view_127 = None
        permute_64: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        addmm_34: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_238, view_128, permute_64);  convert_element_type_238 = None
        view_129: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [64, 512, 256]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_19: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_19: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_17: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_19, 0.1);  convert_element_type_default_19 = None
        mul_83: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_129);  view_129 = None
        mul_84: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, add_45);  mul_84 = add_45 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_22: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_48: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_17: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_47, getitem_23);  add_47 = getitem_23 = None
        mul_85: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_86: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, primals_100)
        add_49: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, primals_101);  mul_86 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_243: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        convert_element_type_244: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_245: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16)
        view_130: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [32768, 256]);  convert_element_type_245 = None
        permute_65: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_35: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_243, view_130, permute_65);  convert_element_type_243 = None
        view_131: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_249: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        mul_87: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_249, 0.5)
        mul_88: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_249, 0.7071067811865476);  convert_element_type_249 = None
        erf_5: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_89: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, add_50);  mul_87 = add_50 = None
        convert_element_type_250: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_251: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_252: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        view_132: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_250, [32768, 1024]);  convert_element_type_250 = None
        permute_66: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        addmm_36: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_251, view_132, permute_66);  convert_element_type_251 = None
        view_133: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [64, 512, 256]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_18: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_18: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_18: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_18, 0.1);  convert_element_type_default_18 = None
        mul_90: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, view_133);  view_133 = None
        mul_91: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, add_49);  mul_91 = add_49 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_24: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_18: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_51, getitem_25);  add_51 = getitem_25 = None
        mul_92: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_93: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, primals_106)
        add_53: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, primals_107);  mul_93 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_256: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_257: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_258: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16)
        view_134: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_258, [32768, 256]);  convert_element_type_258 = None
        permute_67: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_257, [1, 0]);  convert_element_type_257 = None
        addmm_37: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_256, view_134, permute_67);  convert_element_type_256 = None
        view_135: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [64, 512, 256]);  addmm_37 = None
        view_136: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [64, 512, -1, 64]);  view_135 = None
        permute_68: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_262: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        convert_element_type_263: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        permute_69: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_263, [1, 0]);  convert_element_type_263 = None
        addmm_38: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_262, view_134, permute_69);  convert_element_type_262 = None
        view_138: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [64, 512, 256]);  addmm_38 = None
        view_139: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [64, 512, -1, 64]);  view_138 = None
        permute_70: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_268: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_269: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        permute_71: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_269, [1, 0]);  convert_element_type_269 = None
        addmm_39: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_268, view_134, permute_71);  convert_element_type_268 = None
        view_141: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [64, 512, 256]);  addmm_39 = None
        view_142: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_141, [64, 512, -1, 64]);  view_141 = None
        permute_72: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_94: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_68, 0.3535533905932738);  permute_68 = None
        permute_73: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        mul_95: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_73, 0.3535533905932738);  permute_73 = None
        expand_27: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_94, [64, 4, 512, 64]);  mul_94 = None
        clone_24: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_143: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [256, 512, 64]);  clone_24 = None
        expand_28: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_95, [64, 4, 64, 512]);  mul_95 = None
        clone_25: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_144: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [256, 64, 512]);  clone_25 = None
        bmm_12: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_143, view_144)
        view_145: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [64, 4, 512, 512]);  bmm_12 = None
        convert_element_type_276: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_145, torch.float32)
        amax_6: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_276, [-1], True)
        sub_19: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_276, amax_6);  convert_element_type_276 = amax_6 = None
        exp_6: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_277: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        eq_6: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_145, -inf);  view_145 = None
        logical_not_12: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_13: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_13, full_default_1, convert_element_type_277);  logical_not_13 = convert_element_type_277 = None
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_17: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_17: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_19: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_17, 0.1);  convert_element_type_default_17 = None
        mul_96: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, where_13)
        mul_97: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        expand_29: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_97, [64, 4, 512, 512]);  mul_97 = None
        view_146: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [256, 512, 512]);  expand_29 = None
        expand_30: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_72, [64, 4, 512, 64]);  permute_72 = None
        clone_26: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_147: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [256, 512, 64]);  clone_26 = None
        bmm_13: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_146, view_147)
        view_148: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [64, 4, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_27: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_149: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [64, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_280: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_281: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        view_150: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [32768, 256]);  view_149 = None
        permute_75: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        addmm_40: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_280, view_150, permute_75);  convert_element_type_280 = None
        view_151: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [64, 512, 256]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_16: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_16: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_20: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_16, 0.1);  convert_element_type_default_16 = None
        mul_98: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, view_151);  view_151 = None
        mul_99: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, add_53);  mul_99 = add_53 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_26: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_56: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_20: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_55, getitem_27);  add_55 = getitem_27 = None
        mul_100: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_101: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, primals_116)
        add_57: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, primals_117);  mul_101 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_285: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        convert_element_type_286: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_287: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16)
        view_152: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_287, [32768, 256]);  convert_element_type_287 = None
        permute_76: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_286, [1, 0]);  convert_element_type_286 = None
        addmm_41: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_285, view_152, permute_76);  convert_element_type_285 = None
        view_153: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_291: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_153, torch.float32);  view_153 = None
        mul_102: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_291, 0.5)
        mul_103: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_291, 0.7071067811865476);  convert_element_type_291 = None
        erf_6: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_104: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, add_58);  mul_102 = add_58 = None
        convert_element_type_292: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_293: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_294: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        view_154: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_292, [32768, 1024]);  convert_element_type_292 = None
        permute_77: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_294, [1, 0]);  convert_element_type_294 = None
        addmm_42: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_293, view_154, permute_77);  convert_element_type_293 = None
        view_155: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [64, 512, 256]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_15: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_15: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_21: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_15, 0.1);  convert_element_type_default_15 = None
        mul_105: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_155);  view_155 = None
        mul_106: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, add_57);  mul_106 = add_57 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_28: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_21: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_59, getitem_29);  add_59 = getitem_29 = None
        mul_107: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_108: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, primals_122)
        add_61: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_108, primals_123);  mul_108 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_298: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_299: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_300: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_156: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_300, [32768, 256]);  convert_element_type_300 = None
        permute_78: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_299, [1, 0]);  convert_element_type_299 = None
        addmm_43: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_298, view_156, permute_78);  convert_element_type_298 = None
        view_157: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [64, 512, 256]);  addmm_43 = None
        view_158: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_157, [64, 512, -1, 64]);  view_157 = None
        permute_79: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_304: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_305: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        permute_80: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_305, [1, 0]);  convert_element_type_305 = None
        addmm_44: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_304, view_156, permute_80);  convert_element_type_304 = None
        view_160: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [64, 512, 256]);  addmm_44 = None
        view_161: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_160, [64, 512, -1, 64]);  view_160 = None
        permute_81: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_310: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convert_element_type_311: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        permute_82: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_311, [1, 0]);  convert_element_type_311 = None
        addmm_45: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_310, view_156, permute_82);  convert_element_type_310 = None
        view_163: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [64, 512, 256]);  addmm_45 = None
        view_164: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_163, [64, 512, -1, 64]);  view_163 = None
        permute_83: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_109: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_79, 0.3535533905932738);  permute_79 = None
        permute_84: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        mul_110: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
        expand_31: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_109, [64, 4, 512, 64]);  mul_109 = None
        clone_28: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_165: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [256, 512, 64]);  clone_28 = None
        expand_32: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_110, [64, 4, 64, 512]);  mul_110 = None
        clone_29: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_166: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [256, 64, 512]);  clone_29 = None
        bmm_14: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_165, view_166)
        view_167: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [64, 4, 512, 512]);  bmm_14 = None
        convert_element_type_318: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.float32)
        amax_7: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_318, [-1], True)
        sub_22: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_318, amax_7);  convert_element_type_318 = amax_7 = None
        exp_7: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_319: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        eq_7: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_167, -inf);  view_167 = None
        logical_not_14: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_15: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_15, full_default_1, convert_element_type_319);  logical_not_15 = convert_element_type_319 = None
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_14: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_14: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_22: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_14, 0.1);  convert_element_type_default_14 = None
        mul_111: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, where_15)
        mul_112: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        expand_33: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_112, [64, 4, 512, 512]);  mul_112 = None
        view_168: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [256, 512, 512]);  expand_33 = None
        expand_34: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_83, [64, 4, 512, 64]);  permute_83 = None
        clone_30: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_169: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [256, 512, 64]);  clone_30 = None
        bmm_15: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_168, view_169)
        view_170: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [64, 4, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_31: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_171: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [64, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_322: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_323: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        view_172: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [32768, 256]);  view_171 = None
        permute_86: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        addmm_46: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_322, view_172, permute_86);  convert_element_type_322 = None
        view_173: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [64, 512, 256]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_13: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_13: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_23: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_13, 0.1);  convert_element_type_default_13 = None
        mul_113: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_173);  view_173 = None
        mul_114: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, add_61);  mul_114 = add_61 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_30: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_64: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_23: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_63, getitem_31);  add_63 = getitem_31 = None
        mul_115: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_116: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, primals_132)
        add_65: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, primals_133);  mul_116 = primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_327: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        convert_element_type_328: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_329: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        view_174: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_329, [32768, 256]);  convert_element_type_329 = None
        permute_87: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_328, [1, 0]);  convert_element_type_328 = None
        addmm_47: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_327, view_174, permute_87);  convert_element_type_327 = None
        view_175: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_333: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        mul_117: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_333, 0.5)
        mul_118: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_333, 0.7071067811865476);  convert_element_type_333 = None
        erf_7: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_119: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, add_66);  mul_117 = add_66 = None
        convert_element_type_334: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_335: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_336: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        view_176: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_334, [32768, 1024]);  convert_element_type_334 = None
        permute_88: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_336, [1, 0]);  convert_element_type_336 = None
        addmm_48: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_335, view_176, permute_88);  convert_element_type_335 = None
        view_177: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [64, 512, 256]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_12: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_12: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_24: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_12, 0.1);  convert_element_type_default_12 = None
        mul_120: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, view_177);  view_177 = None
        mul_121: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, add_65);  mul_121 = add_65 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_32: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_24: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_67, getitem_33);  add_67 = getitem_33 = None
        mul_122: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_123: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, primals_138)
        add_69: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, primals_139);  mul_123 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_340: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_341: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convert_element_type_342: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16)
        view_178: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_342, [32768, 256]);  convert_element_type_342 = None
        permute_89: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_341, [1, 0]);  convert_element_type_341 = None
        addmm_49: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_340, view_178, permute_89);  convert_element_type_340 = None
        view_179: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [64, 512, 256]);  addmm_49 = None
        view_180: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_179, [64, 512, -1, 64]);  view_179 = None
        permute_90: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_346: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        convert_element_type_347: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        permute_91: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_347, [1, 0]);  convert_element_type_347 = None
        addmm_50: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_346, view_178, permute_91);  convert_element_type_346 = None
        view_182: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [64, 512, 256]);  addmm_50 = None
        view_183: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [64, 512, -1, 64]);  view_182 = None
        permute_92: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_352: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        convert_element_type_353: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        permute_93: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_353, [1, 0]);  convert_element_type_353 = None
        addmm_51: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_352, view_178, permute_93);  convert_element_type_352 = None
        view_185: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [64, 512, 256]);  addmm_51 = None
        view_186: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [64, 512, -1, 64]);  view_185 = None
        permute_94: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_124: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_90, 0.3535533905932738);  permute_90 = None
        permute_95: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        mul_125: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_95, 0.3535533905932738);  permute_95 = None
        expand_35: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_124, [64, 4, 512, 64]);  mul_124 = None
        clone_32: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_187: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [256, 512, 64]);  clone_32 = None
        expand_36: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_125, [64, 4, 64, 512]);  mul_125 = None
        clone_33: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_188: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [256, 64, 512]);  clone_33 = None
        bmm_16: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_187, view_188)
        view_189: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [64, 4, 512, 512]);  bmm_16 = None
        convert_element_type_360: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_189, torch.float32)
        amax_8: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_360, [-1], True)
        sub_25: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, amax_8);  convert_element_type_360 = amax_8 = None
        exp_8: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_361: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        eq_8: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_189, -inf);  view_189 = None
        logical_not_16: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_17: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_17, full_default_1, convert_element_type_361);  logical_not_17 = convert_element_type_361 = None
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_11: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        convert_element_type_default_11: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_25: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_11, 0.1);  convert_element_type_default_11 = None
        mul_126: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_25, where_17)
        mul_127: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, 1.1111111111111112);  mul_126 = None
        expand_37: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_127, [64, 4, 512, 512]);  mul_127 = None
        view_190: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [256, 512, 512]);  expand_37 = None
        expand_38: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_94, [64, 4, 512, 64]);  permute_94 = None
        clone_34: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_191: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [256, 512, 64]);  clone_34 = None
        bmm_17: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_190, view_191)
        view_192: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [64, 4, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_35: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [64, 512, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_364: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_365: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        view_194: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_193, [32768, 256]);  view_193 = None
        permute_97: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_365, [1, 0]);  convert_element_type_365 = None
        addmm_52: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_364, view_194, permute_97);  convert_element_type_364 = None
        view_195: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [64, 512, 256]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_10: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        convert_element_type_default_10: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_26: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_10, 0.1);  convert_element_type_default_10 = None
        mul_128: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, view_195);  view_195 = None
        mul_129: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, add_69);  mul_129 = add_69 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_34: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_72: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_26: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_71, getitem_35);  add_71 = getitem_35 = None
        mul_130: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_131: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, primals_148)
        add_73: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, primals_149);  mul_131 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_369: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        convert_element_type_370: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_371: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16)
        view_196: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [32768, 256]);  convert_element_type_371 = None
        permute_98: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_370, [1, 0]);  convert_element_type_370 = None
        addmm_53: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_369, view_196, permute_98);  convert_element_type_369 = None
        view_197: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_375: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_132: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, 0.5)
        mul_133: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, 0.7071067811865476);  convert_element_type_375 = None
        erf_8: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_134: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, add_74);  mul_132 = add_74 = None
        convert_element_type_376: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_377: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_378: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        view_198: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_376, [32768, 1024]);  convert_element_type_376 = None
        permute_99: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_378, [1, 0]);  convert_element_type_378 = None
        addmm_54: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_377, view_198, permute_99);  convert_element_type_377 = None
        view_199: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [64, 512, 256]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_9: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        convert_element_type_default_9: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_27: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_9, 0.1);  convert_element_type_default_9 = None
        mul_135: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, view_199);  view_199 = None
        mul_136: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_136, add_73);  mul_136 = add_73 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_36: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_27: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_75, getitem_37);  add_75 = getitem_37 = None
        mul_137: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_138: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, primals_154)
        add_77: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_138, primals_155);  mul_138 = primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_382: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        convert_element_type_383: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_384: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16)
        view_200: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_384, [32768, 256]);  convert_element_type_384 = None
        permute_100: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None
        addmm_55: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_382, view_200, permute_100);  convert_element_type_382 = None
        view_201: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [64, 512, 256]);  addmm_55 = None
        view_202: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [64, 512, -1, 64]);  view_201 = None
        permute_101: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_388: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_389: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        permute_102: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_389, [1, 0]);  convert_element_type_389 = None
        addmm_56: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_388, view_200, permute_102);  convert_element_type_388 = None
        view_204: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [64, 512, 256]);  addmm_56 = None
        view_205: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_204, [64, 512, -1, 64]);  view_204 = None
        permute_103: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_394: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        convert_element_type_395: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        permute_104: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_395, [1, 0]);  convert_element_type_395 = None
        addmm_57: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_394, view_200, permute_104);  convert_element_type_394 = None
        view_207: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [64, 512, 256]);  addmm_57 = None
        view_208: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_207, [64, 512, -1, 64]);  view_207 = None
        permute_105: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_139: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_101, 0.3535533905932738);  permute_101 = None
        permute_106: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        mul_140: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_106, 0.3535533905932738);  permute_106 = None
        expand_39: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_139, [64, 4, 512, 64]);  mul_139 = None
        clone_36: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_209: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [256, 512, 64]);  clone_36 = None
        expand_40: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_140, [64, 4, 64, 512]);  mul_140 = None
        clone_37: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_210: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [256, 64, 512]);  clone_37 = None
        bmm_18: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_209, view_210)
        view_211: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [64, 4, 512, 512]);  bmm_18 = None
        convert_element_type_402: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_211, torch.float32)
        amax_9: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_402, [-1], True)
        sub_28: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_402, amax_9);  convert_element_type_402 = amax_9 = None
        exp_9: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_403: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        eq_9: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_211, -inf);  view_211 = None
        logical_not_18: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_19: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_19, full_default_1, convert_element_type_403);  logical_not_19 = convert_element_type_403 = None
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_8: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_8: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_28: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_8, 0.1);  convert_element_type_default_8 = None
        mul_141: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_28, where_19)
        mul_142: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        expand_41: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_142, [64, 4, 512, 512]);  mul_142 = None
        view_212: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [256, 512, 512]);  expand_41 = None
        expand_42: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_105, [64, 4, 512, 64]);  permute_105 = None
        clone_38: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_213: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [256, 512, 64]);  clone_38 = None
        bmm_19: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_212, view_213)
        view_214: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [64, 4, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_39: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [64, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_406: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_407: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        view_216: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [32768, 256]);  view_215 = None
        permute_108: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_407, [1, 0]);  convert_element_type_407 = None
        addmm_58: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_406, view_216, permute_108);  convert_element_type_406 = None
        view_217: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [64, 512, 256]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_7: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_7: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_29: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_143: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_217);  view_217 = None
        mul_144: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, add_77);  mul_144 = add_77 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_38: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_80: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_29: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_79, getitem_39);  add_79 = getitem_39 = None
        mul_145: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_146: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, primals_164)
        add_81: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, primals_165);  mul_146 = primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_411: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        convert_element_type_412: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        convert_element_type_413: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16)
        view_218: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_413, [32768, 256]);  convert_element_type_413 = None
        permute_109: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_412, [1, 0]);  convert_element_type_412 = None
        addmm_59: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_411, view_218, permute_109);  convert_element_type_411 = None
        view_219: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_417: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        mul_147: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, 0.5)
        mul_148: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, 0.7071067811865476);  convert_element_type_417 = None
        erf_9: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_149: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, add_82);  mul_147 = add_82 = None
        convert_element_type_418: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_419: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_420: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        view_220: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_418, [32768, 1024]);  convert_element_type_418 = None
        permute_110: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_420, [1, 0]);  convert_element_type_420 = None
        addmm_60: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_419, view_220, permute_110);  convert_element_type_419 = None
        view_221: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [64, 512, 256]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_6: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        convert_element_type_default_6: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_30: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_6, 0.1);  convert_element_type_default_6 = None
        mul_150: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, view_221);  view_221 = None
        mul_151: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, add_81);  mul_151 = add_81 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_40: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_30: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_83, getitem_41);  add_83 = getitem_41 = None
        mul_152: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_153: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, primals_170)
        add_85: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, primals_171);  mul_153 = primals_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_424: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        convert_element_type_425: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convert_element_type_426: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_222: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_426, [32768, 256]);  convert_element_type_426 = None
        permute_111: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_425, [1, 0]);  convert_element_type_425 = None
        addmm_61: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_424, view_222, permute_111);  convert_element_type_424 = None
        view_223: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [64, 512, 256]);  addmm_61 = None
        view_224: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [64, 512, -1, 64]);  view_223 = None
        permute_112: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_430: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convert_element_type_431: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        permute_113: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_431, [1, 0]);  convert_element_type_431 = None
        addmm_62: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_430, view_222, permute_113);  convert_element_type_430 = None
        view_226: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [64, 512, 256]);  addmm_62 = None
        view_227: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_226, [64, 512, -1, 64]);  view_226 = None
        permute_114: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_436: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        convert_element_type_437: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        permute_115: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_437, [1, 0]);  convert_element_type_437 = None
        addmm_63: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_436, view_222, permute_115);  convert_element_type_436 = None
        view_229: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [64, 512, 256]);  addmm_63 = None
        view_230: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_229, [64, 512, -1, 64]);  view_229 = None
        permute_116: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_154: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_112, 0.3535533905932738);  permute_112 = None
        permute_117: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        mul_155: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_117, 0.3535533905932738);  permute_117 = None
        expand_43: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_154, [64, 4, 512, 64]);  mul_154 = None
        clone_40: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_231: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [256, 512, 64]);  clone_40 = None
        expand_44: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_155, [64, 4, 64, 512]);  mul_155 = None
        clone_41: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_232: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [256, 64, 512]);  clone_41 = None
        bmm_20: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_231, view_232)
        view_233: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [64, 4, 512, 512]);  bmm_20 = None
        convert_element_type_444: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_233, torch.float32)
        amax_10: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_444, [-1], True)
        sub_31: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_444, amax_10);  convert_element_type_444 = amax_10 = None
        exp_10: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_445: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        eq_10: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_233, -inf);  view_233 = None
        logical_not_20: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_21: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_21, full_default_1, convert_element_type_445);  logical_not_21 = convert_element_type_445 = None
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_5: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        convert_element_type_default_5: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_31: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_156: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_31, where_21)
        mul_157: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None
        expand_45: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_157, [64, 4, 512, 512]);  mul_157 = None
        view_234: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [256, 512, 512]);  expand_45 = None
        expand_46: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_116, [64, 4, 512, 64]);  permute_116 = None
        clone_42: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_235: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [256, 512, 64]);  clone_42 = None
        bmm_21: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_234, view_235)
        view_236: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [64, 4, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_43: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [64, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_448: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_449: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        view_238: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_237, [32768, 256]);  view_237 = None
        permute_119: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_449, [1, 0]);  convert_element_type_449 = None
        addmm_64: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_448, view_238, permute_119);  convert_element_type_448 = None
        view_239: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [64, 512, 256]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_4: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_4: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_32: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.1);  convert_element_type_default_4 = None
        mul_158: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, view_239);  view_239 = None
        mul_159: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, 1.1111111111111112);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_159, add_85);  mul_159 = add_85 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_42: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_88: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_32: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_87, getitem_43);  add_87 = getitem_43 = None
        mul_160: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_161: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, primals_180)
        add_89: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, primals_181);  mul_161 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_453: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        convert_element_type_454: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_455: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16)
        view_240: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_455, [32768, 256]);  convert_element_type_455 = None
        permute_120: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_454, [1, 0]);  convert_element_type_454 = None
        addmm_65: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_453, view_240, permute_120);  convert_element_type_453 = None
        view_241: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_459: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        mul_162: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, 0.5)
        mul_163: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, 0.7071067811865476);  convert_element_type_459 = None
        erf_10: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_164: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, add_90);  mul_162 = add_90 = None
        convert_element_type_460: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_461: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_462: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        view_242: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_460, [32768, 1024]);  convert_element_type_460 = None
        permute_121: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_462, [1, 0]);  convert_element_type_462 = None
        addmm_66: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_461, view_242, permute_121);  convert_element_type_461 = None
        view_243: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [64, 512, 256]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_3: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        convert_element_type_default_3: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_33: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_3, 0.1);  convert_element_type_default_3 = None
        mul_165: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_243);  view_243 = None
        mul_166: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_166, add_89);  mul_166 = add_89 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_44: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_33: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_91, getitem_45);  add_91 = getitem_45 = None
        mul_167: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_168: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, primals_186)
        add_93: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_168, primals_187);  mul_168 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_466: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        convert_element_type_467: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_468: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        view_244: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_468, [32768, 256]);  convert_element_type_468 = None
        permute_122: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_467, [1, 0]);  convert_element_type_467 = None
        addmm_67: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_466, view_244, permute_122);  convert_element_type_466 = None
        view_245: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [64, 512, 256]);  addmm_67 = None
        view_246: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [64, 512, -1, 64]);  view_245 = None
        permute_123: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_472: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        convert_element_type_473: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        permute_124: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_473, [1, 0]);  convert_element_type_473 = None
        addmm_68: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_472, view_244, permute_124);  convert_element_type_472 = None
        view_248: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [64, 512, 256]);  addmm_68 = None
        view_249: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_248, [64, 512, -1, 64]);  view_248 = None
        permute_125: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        convert_element_type_478: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        convert_element_type_479: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        permute_126: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_479, [1, 0]);  convert_element_type_479 = None
        addmm_69: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_478, view_244, permute_126);  convert_element_type_478 = None
        view_251: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [64, 512, 256]);  addmm_69 = None
        view_252: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [64, 512, -1, 64]);  view_251 = None
        permute_127: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_169: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_123, 0.3535533905932738);  permute_123 = None
        permute_128: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        mul_170: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_128, 0.3535533905932738);  permute_128 = None
        expand_47: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_169, [64, 4, 512, 64]);  mul_169 = None
        clone_44: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_253: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [256, 512, 64]);  clone_44 = None
        expand_48: "bf16[64, 4, 64, 512][131072, 64, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_170, [64, 4, 64, 512]);  mul_170 = None
        clone_45: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_254: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [256, 64, 512]);  clone_45 = None
        bmm_22: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_253, view_254)
        view_255: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [64, 4, 512, 512]);  bmm_22 = None
        convert_element_type_486: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_255, torch.float32)
        amax_11: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_486, [-1], True)
        sub_34: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_486, amax_11);  convert_element_type_486 = amax_11 = None
        exp_11: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_487: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        eq_11: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_255, -inf);  view_255 = None
        logical_not_22: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_23: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_23, full_default_1, convert_element_type_487);  logical_not_23 = full_default_1 = convert_element_type_487 = None
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_2: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        convert_element_type_default_2: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_34: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_171: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_34, where_23)
        mul_172: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        expand_49: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_172, [64, 4, 512, 512]);  mul_172 = None
        view_256: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [256, 512, 512]);  expand_49 = None
        expand_50: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.expand.default(permute_127, [64, 4, 512, 64]);  permute_127 = None
        clone_46: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_257: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [256, 512, 64]);  clone_46 = None
        bmm_23: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_256, view_257)
        view_258: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [64, 4, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_47: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_259: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [64, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_490: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        convert_element_type_491: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        view_260: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_259, [32768, 256]);  view_259 = None
        permute_130: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_491, [1, 0]);  convert_element_type_491 = None
        addmm_70: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_490, view_260, permute_130);  convert_element_type_490 = None
        view_261: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [64, 512, 256]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_1: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_default_1: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_35: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_173: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_261);  view_261 = None
        mul_174: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, add_93);  mul_174 = add_93 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_46: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_96: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_35: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_95, getitem_47);  add_95 = getitem_47 = None
        mul_175: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_176: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, primals_196)
        add_97: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, primals_197);  mul_176 = primals_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_495: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convert_element_type_496: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convert_element_type_497: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16)
        view_262: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_497, [32768, 256]);  convert_element_type_497 = None
        permute_131: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_496, [1, 0]);  convert_element_type_496 = None
        addmm_71: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_495, view_262, permute_131);  convert_element_type_495 = None
        view_263: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [64, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_501: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_177: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_501, 0.5)
        mul_178: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_501, 0.7071067811865476);  convert_element_type_501 = None
        erf_11: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_179: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, add_98);  mul_177 = add_98 = None
        convert_element_type_502: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_503: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_504: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        view_264: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_502, [32768, 1024]);  convert_element_type_502 = None
        permute_132: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_504, [1, 0]);  convert_element_type_504 = None
        addmm_72: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_503, view_264, permute_132);  convert_element_type_503 = None
        view_265: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [64, 512, 256]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 512, 256], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_default: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_36: "b8[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_180: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, view_265);  view_265 = None
        mul_181: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, add_97);  mul_181 = add_97 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_48: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_36: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_99, getitem_49);  add_99 = getitem_49 = None
        mul_182: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_183: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, primals_202)
        add_101: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, primals_203);  mul_183 = primals_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        convert_element_type_508: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        convert_element_type_509: "bf16[128, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        convert_element_type_510: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None
        view_266: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_510, [32768, 256]);  convert_element_type_510 = None
        permute_133: "bf16[256, 128][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_509, [1, 0]);  convert_element_type_509 = None
        addmm_73: "bf16[32768, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_508, view_266, permute_133);  convert_element_type_508 = None
        view_267: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [64, 512, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_514: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        mul_184: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, 0.5)
        mul_185: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, 0.7071067811865476);  convert_element_type_514 = None
        erf_12: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = add_102 = None
        convert_element_type_515: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_516: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_515, torch.float32);  convert_element_type_515 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_516, [2], correction = 0, keepdim = True)
        getitem_50: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[64, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_103: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_37: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_516, getitem_51);  convert_element_type_516 = None
        mul_187: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_188: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, primals_206);  mul_187 = None
        add_104: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, primals_207);  mul_188 = primals_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        convert_element_type_517: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        convert_element_type_518: "bf16[30522, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_519: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None
        view_268: "bf16[32768, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_519, [32768, 128]);  convert_element_type_519 = None
        permute_134: "bf16[128, 30522][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_518, [1, 0]);  convert_element_type_518 = None
        constant_pad_nd_default_2: "bf16[128, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 6, 0, 0])
        full_default_32: "bf16[6][1]cuda:0" = torch.ops.aten.full.default([6], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[30528][1]cuda:0" = torch.ops.aten.cat.default([convert_element_type_517, full_default_32]);  convert_element_type_517 = full_default_32 = None
        addmm_default: "bf16[32768, 30528][30528, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_268, constant_pad_nd_default_2);  cat_default = constant_pad_nd_default_2 = None
        slice_tensor: "bf16[32768, 30522][30528, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6);  addmm_default = None
        view_269: "bf16[64, 512, 30522][15630336, 30528, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [64, 512, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_523: "f32[64, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_269, torch.float32)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[64, 513][513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1], -100.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[64, 512][513, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone_48: "i64[64, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_270: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_523, [-1, 30522]);  convert_element_type_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_271: "i64[32768][1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [-1]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_12: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_270, [1], True)
        sub_38: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = None
        exp_12: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        ne: "b8[32768][1]cuda:0" = torch.ops.aten.ne.Scalar(view_271, -100)
        full_default_24: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[32768][1]cuda:0" = torch.ops.aten.where.self(ne, view_271, full_default_24);  view_271 = full_default_24 = None
        unsqueeze_3: "i64[32768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_3);  sub_39 = unsqueeze_3 = None
        squeeze: "f32[32768][1]cuda:0" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[32768][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[32768][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_25);  neg = full_default_25 = None
        sum_14: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_524: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_12: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_524);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        permute_135: "bf16[30522, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        permute_139: "bf16[128, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_143: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_147: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_16: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_151: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_156: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_157: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
        permute_158: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_159: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_162: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_167: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_172: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_176: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_180: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_18: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_184: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_189: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        permute_190: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_191: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_192: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_195: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_200: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_205: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_209: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_213: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_20: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_217: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_222: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        permute_223: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_224: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_225: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_228: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_233: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_238: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_242: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_246: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_22: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_250: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_255: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        permute_256: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_257: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_258: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_261: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_266: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_271: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_275: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_279: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_24: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 256);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_283: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_288: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_289: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_290: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_291: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_294: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_299: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_304: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_25: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_308: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_312: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_26: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 256);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_316: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_321: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
        permute_322: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_323: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        permute_324: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_327: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_332: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_337: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_27: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_341: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_345: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_28: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_349: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_354: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_355: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_356: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        permute_357: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_360: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_365: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_370: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_29: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_374: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_378: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_30: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_382: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_387: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        permute_388: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_389: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_390: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_393: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_398: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_403: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_31: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_407: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_411: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_32: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_415: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_420: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        permute_421: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        permute_422: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_423: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_426: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_431: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_436: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_33: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_440: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_444: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_34: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_448: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_453: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_454: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_455: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_456: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_459: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_464: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_469: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_35: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_473: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_477: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_481: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_486: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        permute_487: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_488: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_489: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_492: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_497: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_502: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_37: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_506: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_510: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_514: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_519: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_520: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_521: "bf16[256, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_522: "bf16[256, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_525: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_530: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_535: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        permute_539: "bf16[256, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_39: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (div_12, view_269, primals_2, primals_3, primals_8, primals_20, primals_26, primals_36, primals_42, primals_52, primals_58, primals_68, primals_74, primals_84, primals_90, primals_100, primals_106, primals_116, primals_122, primals_132, primals_138, primals_148, primals_154, primals_164, primals_170, primals_180, primals_186, primals_196, primals_202, primals_206, gather, mul, gt, view, view_2, bmm, amax, sum_1, logical_not_1, gt_1, view_18, gt_2, add_7, getitem_3, rsqrt_1, view_20, addmm_5, view_22, gt_3, mul_17, view_24, where_3, gt_4, view_40, gt_5, mul_25, view_42, addmm_11, view_44, gt_6, mul_32, view_46, where_5, gt_7, view_62, gt_8, mul_40, view_64, addmm_17, view_66, gt_9, mul_47, view_68, where_7, gt_10, view_84, gt_11, mul_55, view_86, addmm_23, view_88, gt_12, mul_62, view_90, where_9, gt_13, view_106, gt_14, mul_70, view_108, addmm_29, view_110, gt_15, mul_77, view_112, where_11, gt_16, view_128, gt_17, mul_85, view_130, addmm_35, view_132, gt_18, mul_92, view_134, where_13, gt_19, view_150, gt_20, mul_100, view_152, addmm_41, view_154, gt_21, mul_107, view_156, where_15, gt_22, view_172, gt_23, mul_115, view_174, addmm_47, view_176, gt_24, mul_122, view_178, where_17, gt_25, view_194, gt_26, mul_130, view_196, addmm_53, view_198, gt_27, mul_137, view_200, where_19, gt_28, view_216, gt_29, mul_145, view_218, addmm_59, view_220, gt_30, mul_152, view_222, where_21, gt_31, view_238, gt_32, mul_160, view_240, addmm_65, view_242, gt_33, mul_167, view_244, where_23, gt_34, view_260, gt_35, mul_175, view_262, addmm_71, view_264, gt_36, mul_182, view_266, addmm_73, getitem_51, rsqrt_25, view_268, view_269, constant_pad_nd, amax_12, log, convert_element_type_524, permute_135, permute_139, div_15, permute_143, permute_147, div_16, permute_151, permute_156, permute_157, permute_158, permute_159, permute_162, permute_167, permute_172, div_17, permute_176, permute_180, div_18, permute_184, permute_189, permute_190, permute_191, permute_192, permute_195, permute_200, permute_205, div_19, permute_209, permute_213, div_20, permute_217, permute_222, permute_223, permute_224, permute_225, permute_228, permute_233, permute_238, div_21, permute_242, permute_246, div_22, permute_250, permute_255, permute_256, permute_257, permute_258, permute_261, permute_266, permute_271, div_23, permute_275, permute_279, div_24, permute_283, permute_288, permute_289, permute_290, permute_291, permute_294, permute_299, permute_304, div_25, permute_308, permute_312, div_26, permute_316, permute_321, permute_322, permute_323, permute_324, permute_327, permute_332, permute_337, div_27, permute_341, permute_345, div_28, permute_349, permute_354, permute_355, permute_356, permute_357, permute_360, permute_365, permute_370, div_29, permute_374, permute_378, div_30, permute_382, permute_387, permute_388, permute_389, permute_390, permute_393, permute_398, permute_403, div_31, permute_407, permute_411, div_32, permute_415, permute_420, permute_421, permute_422, permute_423, permute_426, permute_431, permute_436, div_33, permute_440, permute_444, div_34, permute_448, permute_453, permute_454, permute_455, permute_456, permute_459, permute_464, permute_469, div_35, permute_473, permute_477, div_36, permute_481, permute_486, permute_487, permute_488, permute_489, permute_492, permute_497, permute_502, div_37, permute_506, permute_510, permute_514, permute_519, permute_520, permute_521, permute_522, permute_525, permute_530, permute_535, permute_539, div_39)
