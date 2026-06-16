class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512][512, 1]cuda:0", primals_2: "i64[1, 512][512, 1]cuda:0", primals_3: "f32[30522, 768][768, 1]cuda:0", primals_4: "i64[1, 512][512, 1]cuda:0", primals_5: "f32[512, 768][768, 1]cuda:0", primals_6: "f32[2, 768][768, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768][1]cuda:0", primals_9: "f32[384, 768][768, 1]cuda:0", primals_10: "f32[384][1]cuda:0", primals_11: "f32[384, 768][768, 1]cuda:0", primals_12: "f32[384][1]cuda:0", primals_13: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_14: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_15: "f32[384, 1][1, 1]cuda:0", primals_16: "f32[384, 768][768, 1]cuda:0", primals_17: "f32[384][1]cuda:0", primals_18: "f32[54, 384][384, 1]cuda:0", primals_19: "f32[54][1]cuda:0", primals_20: "f32[384, 768][768, 1]cuda:0", primals_21: "f32[384][1]cuda:0", primals_22: "f32[768, 768][768, 1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_24: "f32[768][1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_26: "f32[3072, 768][768, 1]cuda:0", primals_27: "f32[3072][1]cuda:0", primals_28: "f32[768, 3072][3072, 1]cuda:0", primals_29: "f32[768][1]cuda:0", primals_30: "f32[768][1]cuda:0", primals_31: "f32[768][1]cuda:0", primals_32: "f32[384, 768][768, 1]cuda:0", primals_33: "f32[384][1]cuda:0", primals_34: "f32[384, 768][768, 1]cuda:0", primals_35: "f32[384][1]cuda:0", primals_36: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_37: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_38: "f32[384, 1][1, 1]cuda:0", primals_39: "f32[384, 768][768, 1]cuda:0", primals_40: "f32[384][1]cuda:0", primals_41: "f32[54, 384][384, 1]cuda:0", primals_42: "f32[54][1]cuda:0", primals_43: "f32[384, 768][768, 1]cuda:0", primals_44: "f32[384][1]cuda:0", primals_45: "f32[768, 768][768, 1]cuda:0", primals_46: "f32[768][1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[3072, 768][768, 1]cuda:0", primals_50: "f32[3072][1]cuda:0", primals_51: "f32[768, 3072][3072, 1]cuda:0", primals_52: "f32[768][1]cuda:0", primals_53: "f32[768][1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[384, 768][768, 1]cuda:0", primals_56: "f32[384][1]cuda:0", primals_57: "f32[384, 768][768, 1]cuda:0", primals_58: "f32[384][1]cuda:0", primals_59: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_60: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_61: "f32[384, 1][1, 1]cuda:0", primals_62: "f32[384, 768][768, 1]cuda:0", primals_63: "f32[384][1]cuda:0", primals_64: "f32[54, 384][384, 1]cuda:0", primals_65: "f32[54][1]cuda:0", primals_66: "f32[384, 768][768, 1]cuda:0", primals_67: "f32[384][1]cuda:0", primals_68: "f32[768, 768][768, 1]cuda:0", primals_69: "f32[768][1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_72: "f32[3072, 768][768, 1]cuda:0", primals_73: "f32[3072][1]cuda:0", primals_74: "f32[768, 3072][3072, 1]cuda:0", primals_75: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_77: "f32[768][1]cuda:0", primals_78: "f32[384, 768][768, 1]cuda:0", primals_79: "f32[384][1]cuda:0", primals_80: "f32[384, 768][768, 1]cuda:0", primals_81: "f32[384][1]cuda:0", primals_82: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_83: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_84: "f32[384, 1][1, 1]cuda:0", primals_85: "f32[384, 768][768, 1]cuda:0", primals_86: "f32[384][1]cuda:0", primals_87: "f32[54, 384][384, 1]cuda:0", primals_88: "f32[54][1]cuda:0", primals_89: "f32[384, 768][768, 1]cuda:0", primals_90: "f32[384][1]cuda:0", primals_91: "f32[768, 768][768, 1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_93: "f32[768][1]cuda:0", primals_94: "f32[768][1]cuda:0", primals_95: "f32[3072, 768][768, 1]cuda:0", primals_96: "f32[3072][1]cuda:0", primals_97: "f32[768, 3072][3072, 1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[768][1]cuda:0", primals_100: "f32[768][1]cuda:0", primals_101: "f32[384, 768][768, 1]cuda:0", primals_102: "f32[384][1]cuda:0", primals_103: "f32[384, 768][768, 1]cuda:0", primals_104: "f32[384][1]cuda:0", primals_105: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_106: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_107: "f32[384, 1][1, 1]cuda:0", primals_108: "f32[384, 768][768, 1]cuda:0", primals_109: "f32[384][1]cuda:0", primals_110: "f32[54, 384][384, 1]cuda:0", primals_111: "f32[54][1]cuda:0", primals_112: "f32[384, 768][768, 1]cuda:0", primals_113: "f32[384][1]cuda:0", primals_114: "f32[768, 768][768, 1]cuda:0", primals_115: "f32[768][1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_117: "f32[768][1]cuda:0", primals_118: "f32[3072, 768][768, 1]cuda:0", primals_119: "f32[3072][1]cuda:0", primals_120: "f32[768, 3072][3072, 1]cuda:0", primals_121: "f32[768][1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_123: "f32[768][1]cuda:0", primals_124: "f32[384, 768][768, 1]cuda:0", primals_125: "f32[384][1]cuda:0", primals_126: "f32[384, 768][768, 1]cuda:0", primals_127: "f32[384][1]cuda:0", primals_128: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_129: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_130: "f32[384, 1][1, 1]cuda:0", primals_131: "f32[384, 768][768, 1]cuda:0", primals_132: "f32[384][1]cuda:0", primals_133: "f32[54, 384][384, 1]cuda:0", primals_134: "f32[54][1]cuda:0", primals_135: "f32[384, 768][768, 1]cuda:0", primals_136: "f32[384][1]cuda:0", primals_137: "f32[768, 768][768, 1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[3072, 768][768, 1]cuda:0", primals_142: "f32[3072][1]cuda:0", primals_143: "f32[768, 3072][3072, 1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[384, 768][768, 1]cuda:0", primals_148: "f32[384][1]cuda:0", primals_149: "f32[384, 768][768, 1]cuda:0", primals_150: "f32[384][1]cuda:0", primals_151: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_152: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_153: "f32[384, 1][1, 1]cuda:0", primals_154: "f32[384, 768][768, 1]cuda:0", primals_155: "f32[384][1]cuda:0", primals_156: "f32[54, 384][384, 1]cuda:0", primals_157: "f32[54][1]cuda:0", primals_158: "f32[384, 768][768, 1]cuda:0", primals_159: "f32[384][1]cuda:0", primals_160: "f32[768, 768][768, 1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_162: "f32[768][1]cuda:0", primals_163: "f32[768][1]cuda:0", primals_164: "f32[3072, 768][768, 1]cuda:0", primals_165: "f32[3072][1]cuda:0", primals_166: "f32[768, 3072][3072, 1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_169: "f32[768][1]cuda:0", primals_170: "f32[384, 768][768, 1]cuda:0", primals_171: "f32[384][1]cuda:0", primals_172: "f32[384, 768][768, 1]cuda:0", primals_173: "f32[384][1]cuda:0", primals_174: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_175: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_176: "f32[384, 1][1, 1]cuda:0", primals_177: "f32[384, 768][768, 1]cuda:0", primals_178: "f32[384][1]cuda:0", primals_179: "f32[54, 384][384, 1]cuda:0", primals_180: "f32[54][1]cuda:0", primals_181: "f32[384, 768][768, 1]cuda:0", primals_182: "f32[384][1]cuda:0", primals_183: "f32[768, 768][768, 1]cuda:0", primals_184: "f32[768][1]cuda:0", primals_185: "f32[768][1]cuda:0", primals_186: "f32[768][1]cuda:0", primals_187: "f32[3072, 768][768, 1]cuda:0", primals_188: "f32[3072][1]cuda:0", primals_189: "f32[768, 3072][3072, 1]cuda:0", primals_190: "f32[768][1]cuda:0", primals_191: "f32[768][1]cuda:0", primals_192: "f32[768][1]cuda:0", primals_193: "f32[384, 768][768, 1]cuda:0", primals_194: "f32[384][1]cuda:0", primals_195: "f32[384, 768][768, 1]cuda:0", primals_196: "f32[384][1]cuda:0", primals_197: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_198: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_199: "f32[384, 1][1, 1]cuda:0", primals_200: "f32[384, 768][768, 1]cuda:0", primals_201: "f32[384][1]cuda:0", primals_202: "f32[54, 384][384, 1]cuda:0", primals_203: "f32[54][1]cuda:0", primals_204: "f32[384, 768][768, 1]cuda:0", primals_205: "f32[384][1]cuda:0", primals_206: "f32[768, 768][768, 1]cuda:0", primals_207: "f32[768][1]cuda:0", primals_208: "f32[768][1]cuda:0", primals_209: "f32[768][1]cuda:0", primals_210: "f32[3072, 768][768, 1]cuda:0", primals_211: "f32[3072][1]cuda:0", primals_212: "f32[768, 3072][3072, 1]cuda:0", primals_213: "f32[768][1]cuda:0", primals_214: "f32[768][1]cuda:0", primals_215: "f32[768][1]cuda:0", primals_216: "f32[384, 768][768, 1]cuda:0", primals_217: "f32[384][1]cuda:0", primals_218: "f32[384, 768][768, 1]cuda:0", primals_219: "f32[384][1]cuda:0", primals_220: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_221: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_222: "f32[384, 1][1, 1]cuda:0", primals_223: "f32[384, 768][768, 1]cuda:0", primals_224: "f32[384][1]cuda:0", primals_225: "f32[54, 384][384, 1]cuda:0", primals_226: "f32[54][1]cuda:0", primals_227: "f32[384, 768][768, 1]cuda:0", primals_228: "f32[384][1]cuda:0", primals_229: "f32[768, 768][768, 1]cuda:0", primals_230: "f32[768][1]cuda:0", primals_231: "f32[768][1]cuda:0", primals_232: "f32[768][1]cuda:0", primals_233: "f32[3072, 768][768, 1]cuda:0", primals_234: "f32[3072][1]cuda:0", primals_235: "f32[768, 3072][3072, 1]cuda:0", primals_236: "f32[768][1]cuda:0", primals_237: "f32[768][1]cuda:0", primals_238: "f32[768][1]cuda:0", primals_239: "f32[384, 768][768, 1]cuda:0", primals_240: "f32[384][1]cuda:0", primals_241: "f32[384, 768][768, 1]cuda:0", primals_242: "f32[384][1]cuda:0", primals_243: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_244: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_245: "f32[384, 1][1, 1]cuda:0", primals_246: "f32[384, 768][768, 1]cuda:0", primals_247: "f32[384][1]cuda:0", primals_248: "f32[54, 384][384, 1]cuda:0", primals_249: "f32[54][1]cuda:0", primals_250: "f32[384, 768][768, 1]cuda:0", primals_251: "f32[384][1]cuda:0", primals_252: "f32[768, 768][768, 1]cuda:0", primals_253: "f32[768][1]cuda:0", primals_254: "f32[768][1]cuda:0", primals_255: "f32[768][1]cuda:0", primals_256: "f32[3072, 768][768, 1]cuda:0", primals_257: "f32[3072][1]cuda:0", primals_258: "f32[768, 3072][3072, 1]cuda:0", primals_259: "f32[768][1]cuda:0", primals_260: "f32[768][1]cuda:0", primals_261: "f32[768][1]cuda:0", primals_262: "f32[384, 768][768, 1]cuda:0", primals_263: "f32[384][1]cuda:0", primals_264: "f32[384, 768][768, 1]cuda:0", primals_265: "f32[384][1]cuda:0", primals_266: "f32[768, 1, 9][9, 9, 1]cuda:0", primals_267: "f32[384, 768, 1][768, 1, 1]cuda:0", primals_268: "f32[384, 1][1, 1]cuda:0", primals_269: "f32[384, 768][768, 1]cuda:0", primals_270: "f32[384][1]cuda:0", primals_271: "f32[54, 384][384, 1]cuda:0", primals_272: "f32[54][1]cuda:0", primals_273: "f32[384, 768][768, 1]cuda:0", primals_274: "f32[384][1]cuda:0", primals_275: "f32[768, 768][768, 1]cuda:0", primals_276: "f32[768][1]cuda:0", primals_277: "f32[768][1]cuda:0", primals_278: "f32[768][1]cuda:0", primals_279: "f32[3072, 768][768, 1]cuda:0", primals_280: "f32[3072][1]cuda:0", primals_281: "f32[768, 3072][3072, 1]cuda:0", primals_282: "f32[768][1]cuda:0", primals_283: "f32[768][1]cuda:0", primals_284: "f32[768][1]cuda:0", primals_285: "f32[768, 768][768, 1]cuda:0", primals_286: "f32[768][1]cuda:0", primals_287: "f32[768][1]cuda:0", primals_288: "f32[768][1]cuda:0", primals_289: "f32[30522][1]cuda:0", primals_290: "i64[32, 512][512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:628 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[32, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(primals_2, [32, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:99 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_3, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:100 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[1, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_5, primals_4);  primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:101 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_6, expand);  primals_6 = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, primals_7)
        add_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, primals_8);  mul_2 = primals_8 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[25][1]cuda:0" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:105 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_4: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_1: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_2: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 768]);  convert_element_type_2 = None
        permute: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 512, 384]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_6: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_7: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        permute_1: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_1);  convert_element_type_6 = None
        view_3: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 384]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_2: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(mul_4, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_12: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_13: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_2, torch.bfloat16);  permute_2 = None
        convolution: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_13, convert_element_type_12, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_14: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_1: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution, convert_element_type_14, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_4: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_1, primals_15)
        convert_element_type_15: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_16: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_17: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        permute_4: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        addmm_2: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_16, view, permute_4);  convert_element_type_16 = None
        view_5: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_6: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_7: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_8: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [32, 512, -1, 64]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_8: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1]);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_5: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_8, view_5);  permute_8 = view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_22: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16)
        convert_element_type_23: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        permute_9: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        clone: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_5, memory_format = torch.contiguous_format);  mul_5 = None
        view_9: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [16384, 384]);  clone = None
        mm: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_9, permute_9)
        view_10: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 512, 54])
        add_5: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_10, convert_element_type_22);  view_10 = convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_11: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [-1, 9, 1]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_26: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_26, [1], True)
        sub_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax);  convert_element_type_26 = None
        exp: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_27: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_28: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_10: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        addmm_3: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_27, view, permute_10);  convert_element_type_27 = None
        view_13: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 384]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_11: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        clone_1: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        unsqueeze_2: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_1, -1);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add_6: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        full_default_1: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        constant_pad_nd: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 0, 4, 4], 0.0);  unsqueeze_2 = None
        unsqueeze_7: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
        unsqueeze_8: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
        index: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd = None
        permute_12: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_15: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_12, [32, 3456, 512]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_13: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        view_16: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_13, [32, 512, 384, 9]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_2: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format);  view_16 = None
        view_17: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [98304, 64, 9]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_33: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_1: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_17, [98304, 64, 9]);  view_17 = None
        expand_2: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_33, [98304, 9, 1]);  convert_element_type_33 = None
        bmm: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_1, expand_2);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_21: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [-1, 384]);  bmm = None

        # No stacktrace found for following nodes
        permute_default_66: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_default_67: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        permute_default_68: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_66, permute_default_67, permute_default_68, 0.1, scale = 0.125);  permute_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_201: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_11[0]

        # No stacktrace found for following nodes
        getitem_202: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_11[1]
        getitem_203: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_11[6]
        getitem_204: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_11[7];  _scaled_dot_product_flash_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_15: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3])
        clone_6: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_28: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_21, [32, -1, 6, 64]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_6, view_28], 2);  clone_6 = view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [32, 512, 768]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_41: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_42: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        view_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [16384, 768]);  view_29 = None
        permute_16: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        addmm_4: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_41, view_30, permute_16);  convert_element_type_41 = None
        view_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 768]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_35: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_2: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_35, 0.1);  convert_element_type_default_35 = None
        mul_8: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_31);  view_31 = None
        mul_9: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, mul_4);  mul_9 = mul_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_9, getitem_3);  add_9 = getitem_3 = None
        mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
        mul_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_24)
        add_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_25);  mul_11 = primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_46: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_47: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convert_element_type_48: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16)
        view_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_48, [16384, 768]);  convert_element_type_48 = None
        permute_17: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        addmm_5: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_46, view_32, permute_17);  convert_element_type_46 = None
        view_33: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_52: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_12: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, 0.5)
        mul_13: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, 0.7071067811865476);  convert_element_type_52 = None
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_12: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_12);  mul_12 = add_12 = None
        convert_element_type_53: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_54: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_55: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        view_34: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_53, [16384, 3072]);  convert_element_type_53 = None
        permute_18: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        addmm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_54, view_34, permute_18);  convert_element_type_54 = None
        view_35: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_22: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_34: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_3: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_34, 0.1);  convert_element_type_default_34 = None
        mul_15: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_35);  view_35 = None
        mul_16: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, add_11);  mul_16 = add_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_13, getitem_5);  add_13 = getitem_5 = None
        mul_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = None
        mul_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, primals_30)
        add_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, primals_31);  mul_18 = primals_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_59: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_60: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_61: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16)
        view_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [16384, 768]);  convert_element_type_61 = None
        permute_19: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_60, [1, 0]);  convert_element_type_60 = None
        addmm_7: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_59, view_36, permute_19);  convert_element_type_59 = None
        view_37: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 384]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_65: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_66: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        permute_20: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_8: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_65, view_36, permute_20);  convert_element_type_65 = None
        view_39: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 384]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_21: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_15, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_71: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convert_element_type_72: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_21, torch.bfloat16);  permute_21 = None
        convolution_2: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_72, convert_element_type_71, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_73: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convolution_3: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_2, convert_element_type_73, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_16: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_3, primals_38)
        convert_element_type_74: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_75: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convert_element_type_76: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_23: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        addmm_9: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_75, view_36, permute_23);  convert_element_type_75 = None
        view_41: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_42: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_43: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [32, 512, -1, 64]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_44: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [32, 512, -1, 64]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_27: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_74, [0, 2, 1]);  convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_19: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_27, view_41);  permute_27 = view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_81: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16)
        convert_element_type_82: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        permute_28: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        clone_7: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_19, memory_format = torch.contiguous_format);  mul_19 = None
        view_45: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [16384, 384]);  clone_7 = None
        mm_1: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_28)
        view_46: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [32, 512, 54])
        add_17: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_46, convert_element_type_81);  view_46 = convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_47: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_17, [-1, 9, 1]);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_85: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        amax_2: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_85, [1], True)
        sub_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, amax_2);  convert_element_type_85 = None
        exp_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
        div_3: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_86: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_87: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_29: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        addmm_10: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_86, view_36, permute_29);  convert_element_type_86 = None
        view_49: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_30: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        clone_8: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        unsqueeze_9: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_8, -1);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_1: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 0, 4, 4], 0.0);  unsqueeze_9 = None
        index_1: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_1, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_1 = None
        permute_31: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        view_51: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_31, [32, 3456, 512]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_32: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        view_52: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_32, [32, 512, 384, 9]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_9: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
        view_53: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [98304, 64, 9]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_92: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        expand_7: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_53, [98304, 64, 9]);  view_53 = None
        expand_8: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_92, [98304, 9, 1]);  convert_element_type_92 = None
        bmm_3: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_7, expand_8);  expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_57: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [-1, 384]);  bmm_3 = None

        # No stacktrace found for following nodes
        permute_default_60: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        permute_default_61: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        permute_default_62: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_60, permute_default_61, permute_default_62, 0.1, scale = 0.125);  permute_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_194: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_10[0]

        # No stacktrace found for following nodes
        getitem_195: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_10[1]
        getitem_196: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_10[6]
        getitem_197: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_10[7];  _scaled_dot_product_flash_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_34: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3])
        clone_13: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_64: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [32, -1, 6, 64]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_1: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_13, view_64], 2);  clone_13 = view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_65: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [32, 512, 768]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_100: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_101: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        view_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [16384, 768]);  view_65 = None
        permute_35: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        addmm_11: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_100, view_66, permute_35);  convert_element_type_100 = None
        view_67: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_33: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_5: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_33, 0.1);  convert_element_type_default_33 = None
        mul_22: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_67);  view_67 = None
        mul_23: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, 1.1111111111111112);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, add_15);  mul_23 = add_15 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_8: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_21, getitem_7);  add_21 = getitem_7 = None
        mul_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = None
        mul_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, primals_47)
        add_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, primals_48);  mul_25 = primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_105: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_106: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_107: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16)
        view_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_107, [16384, 768]);  convert_element_type_107 = None
        permute_36: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_106, [1, 0]);  convert_element_type_106 = None
        addmm_12: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_105, view_68, permute_36);  convert_element_type_105 = None
        view_69: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_111: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_26: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_111, 0.5)
        mul_27: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_111, 0.7071067811865476);  convert_element_type_111 = None
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_24: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_28: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_24);  mul_26 = add_24 = None
        convert_element_type_112: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_113: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_114: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        view_70: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_112, [16384, 3072]);  convert_element_type_112 = None
        permute_37: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        addmm_13: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_113, view_70, permute_37);  convert_element_type_113 = None
        view_71: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_32: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_20, torch.bfloat16);  inductor_random_default_20 = None
        gt_6: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_32, 0.1);  convert_element_type_default_32 = None
        mul_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, view_71);  view_71 = None
        mul_30: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, 1.1111111111111112);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, add_23);  mul_30 = add_23 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_25, getitem_9);  add_25 = getitem_9 = None
        mul_31: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = None
        mul_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, primals_53)
        add_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, primals_54);  mul_32 = primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_118: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_119: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_120: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16)
        view_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [16384, 768]);  convert_element_type_120 = None
        permute_38: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        addmm_14: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_118, view_72, permute_38);  convert_element_type_118 = None
        view_73: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_124: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_125: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        permute_39: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        addmm_15: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_124, view_72, permute_39);  convert_element_type_124 = None
        view_75: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 384]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_40: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_27, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_130: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_131: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_40, torch.bfloat16);  permute_40 = None
        convolution_4: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_131, convert_element_type_130, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_132: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convolution_5: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_4, convert_element_type_132, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_28: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_5, primals_61)
        convert_element_type_133: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_134: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_135: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        permute_42: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        addmm_16: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_134, view_72, permute_42);  convert_element_type_134 = None
        view_77: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_78: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_79: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_80: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [32, 512, -1, 64]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_46: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_133, [0, 2, 1]);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_33: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_46, view_77);  permute_46 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_140: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16)
        convert_element_type_141: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        permute_47: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        clone_14: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_33, memory_format = torch.contiguous_format);  mul_33 = None
        view_81: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [16384, 384]);  clone_14 = None
        mm_2: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_81, permute_47)
        view_82: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 512, 54])
        add_29: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_82, convert_element_type_140);  view_82 = convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_83: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_29, [-1, 9, 1]);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_144: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None
        amax_4: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_144, [1], True)
        sub_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, amax_4);  convert_element_type_144 = None
        exp_4: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_5: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
        div_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_145: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_146: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        permute_48: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        addmm_17: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_145, view_72, permute_48);  convert_element_type_145 = None
        view_85: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 384]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_49: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        clone_15: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        unsqueeze_16: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_15, -1);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_2: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_16, [0, 0, 4, 4], 0.0);  unsqueeze_16 = None
        index_2: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_2, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_2 = None
        permute_50: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
        view_87: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_50, [32, 3456, 512]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_51: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        view_88: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_51, [32, 512, 384, 9]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_16: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_88, memory_format = torch.contiguous_format);  view_88 = None
        view_89: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [98304, 64, 9]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_151: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_13: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_89, [98304, 64, 9]);  view_89 = None
        expand_14: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_151, [98304, 9, 1]);  convert_element_type_151 = None
        bmm_6: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_13, expand_14);  expand_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_93: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [-1, 384]);  bmm_6 = None

        # No stacktrace found for following nodes
        permute_default_54: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        permute_default_55: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        permute_default_56: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_54, permute_default_55, permute_default_56, 0.1, scale = 0.125);  permute_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_187: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_9[0]

        # No stacktrace found for following nodes
        getitem_188: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_9[1]
        getitem_189: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_9[6]
        getitem_190: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_9[7];  _scaled_dot_product_flash_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_53: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3])
        clone_20: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_100: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_93, [32, -1, 6, 64]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_2: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_20, view_100], 2);  clone_20 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_101: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [32, 512, 768]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_159: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_160: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        view_102: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_101, [16384, 768]);  view_101 = None
        permute_54: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        addmm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_159, view_102, permute_54);  convert_element_type_159 = None
        view_103: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_19: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_8: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_31, 0.1);  convert_element_type_default_31 = None
        mul_36: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, view_103);  view_103 = None
        mul_37: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, add_27);  mul_37 = add_27 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_33, getitem_11);  add_33 = getitem_11 = None
        mul_38: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = None
        mul_39: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, primals_70)
        add_35: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, primals_71);  mul_39 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_164: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_165: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_166: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16)
        view_104: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_166, [16384, 768]);  convert_element_type_166 = None
        permute_55: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_165, [1, 0]);  convert_element_type_165 = None
        addmm_19: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_164, view_104, permute_55);  convert_element_type_164 = None
        view_105: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_170: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_40: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, 0.5)
        mul_41: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, 0.7071067811865476);  convert_element_type_170 = None
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_36: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_42: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, add_36);  mul_40 = add_36 = None
        convert_element_type_171: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_172: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_173: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        view_106: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [16384, 3072]);  convert_element_type_171 = None
        permute_56: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        addmm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_172, view_106, permute_56);  convert_element_type_172 = None
        view_107: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_30: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_9: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_30, 0.1);  convert_element_type_default_30 = None
        mul_43: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_107);  view_107 = None
        mul_44: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, add_35);  mul_44 = add_35 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
        mul_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = None
        mul_46: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, primals_76)
        add_39: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, primals_77);  mul_46 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_177: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_178: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_179: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16)
        view_108: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_179, [16384, 768]);  convert_element_type_179 = None
        permute_57: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        addmm_21: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_177, view_108, permute_57);  convert_element_type_177 = None
        view_109: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 384]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_183: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_184: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        permute_58: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_184, [1, 0]);  convert_element_type_184 = None
        addmm_22: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_183, view_108, permute_58);  convert_element_type_183 = None
        view_111: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_59: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_39, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_189: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_190: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_59, torch.bfloat16);  permute_59 = None
        convolution_6: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_190, convert_element_type_189, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_191: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convolution_7: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_6, convert_element_type_191, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_40: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_7, primals_84)
        convert_element_type_192: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_193: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_194: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        permute_61: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_194, [1, 0]);  convert_element_type_194 = None
        addmm_23: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_193, view_108, permute_61);  convert_element_type_193 = None
        view_113: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_114: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_115: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_109, [32, 512, -1, 64]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_116: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_65: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_192, [0, 2, 1]);  convert_element_type_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_47: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_65, view_113);  permute_65 = view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_199: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16)
        convert_element_type_200: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        permute_66: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        clone_21: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_47, memory_format = torch.contiguous_format);  mul_47 = None
        view_117: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [16384, 384]);  clone_21 = None
        mm_3: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_117, permute_66)
        view_118: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [32, 512, 54])
        add_41: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_118, convert_element_type_199);  view_118 = convert_element_type_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_119: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_41, [-1, 9, 1]);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_203: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        amax_6: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_203, [1], True)
        sub_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, amax_6);  convert_element_type_203 = None
        exp_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_7: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
        div_9: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_204: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_205: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        permute_67: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_205, [1, 0]);  convert_element_type_205 = None
        addmm_24: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_204, view_108, permute_67);  convert_element_type_204 = None
        view_121: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 384]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_68: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        clone_22: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        unsqueeze_23: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_22, -1);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_3: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_23, [0, 0, 4, 4], 0.0);  unsqueeze_23 = None
        index_3: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_3, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_3 = None
        permute_69: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
        view_123: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [32, 3456, 512]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_70: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        view_124: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_70, [32, 512, 384, 9]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_23: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
        view_125: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [98304, 64, 9]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_210: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        expand_19: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_125, [98304, 64, 9]);  view_125 = None
        expand_20: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_210, [98304, 9, 1]);  convert_element_type_210 = None
        bmm_9: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_19, expand_20);  expand_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_129: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [-1, 384]);  bmm_9 = None

        # No stacktrace found for following nodes
        permute_default_48: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_default_49: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        permute_default_50: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_48, permute_default_49, permute_default_50, 0.1, scale = 0.125);  permute_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_180: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_8[0]

        # No stacktrace found for following nodes
        getitem_181: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_8[1]
        getitem_182: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_8[6]
        getitem_183: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_8[7];  _scaled_dot_product_flash_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_72: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3])
        clone_27: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_136: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_129, [32, -1, 6, 64]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_3: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_27, view_136], 2);  clone_27 = view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_137: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [32, 512, 768]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_218: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_219: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        view_138: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [16384, 768]);  view_137 = None
        permute_73: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        addmm_25: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_218, view_138, permute_73);  convert_element_type_218 = None
        view_139: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_11: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_29, 0.1);  convert_element_type_default_29 = None
        mul_50: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_139);  view_139 = None
        mul_51: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, add_39);  mul_51 = add_39 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_16: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_45, getitem_15);  add_45 = getitem_15 = None
        mul_52: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = None
        mul_53: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, primals_93)
        add_47: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, primals_94);  mul_53 = primals_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_223: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convert_element_type_224: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_225: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16)
        view_140: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_225, [16384, 768]);  convert_element_type_225 = None
        permute_74: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        addmm_26: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_223, view_140, permute_74);  convert_element_type_223 = None
        view_141: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_229: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_54: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, 0.5)
        mul_55: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, 0.7071067811865476);  convert_element_type_229 = None
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_48: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_56: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_48);  mul_54 = add_48 = None
        convert_element_type_230: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_231: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convert_element_type_232: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        view_142: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_230, [16384, 3072]);  convert_element_type_230 = None
        permute_75: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None
        addmm_27: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_231, view_142, permute_75);  convert_element_type_231 = None
        view_143: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_16: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_28: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_12: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_28, 0.1);  convert_element_type_default_28 = None
        mul_57: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, view_143);  view_143 = None
        mul_58: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, add_47);  mul_58 = add_47 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_49, getitem_17);  add_49 = getitem_17 = None
        mul_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = None
        mul_60: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, primals_99)
        add_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, primals_100);  mul_60 = primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_236: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_237: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_238: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16)
        view_144: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_238, [16384, 768]);  convert_element_type_238 = None
        permute_76: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_237, [1, 0]);  convert_element_type_237 = None
        addmm_28: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_236, view_144, permute_76);  convert_element_type_236 = None
        view_145: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 384]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_242: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convert_element_type_243: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        permute_77: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_243, [1, 0]);  convert_element_type_243 = None
        addmm_29: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_242, view_144, permute_77);  convert_element_type_242 = None
        view_147: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 384]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_78: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_51, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_248: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_249: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_78, torch.bfloat16);  permute_78 = None
        convolution_8: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_249, convert_element_type_248, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_250: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convolution_9: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_8, convert_element_type_250, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_52: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_9, primals_107)
        convert_element_type_251: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_252: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_253: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        permute_80: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_30: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_252, view_144, permute_80);  convert_element_type_252 = None
        view_149: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_150: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_151: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [32, 512, -1, 64]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_152: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [32, 512, -1, 64]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_84: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_251, [0, 2, 1]);  convert_element_type_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_61: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_84, view_149);  permute_84 = view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_258: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16)
        convert_element_type_259: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        permute_85: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        clone_28: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_61, memory_format = torch.contiguous_format);  mul_61 = None
        view_153: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [16384, 384]);  clone_28 = None
        mm_4: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_153, permute_85)
        view_154: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 512, 54])
        add_53: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_154, convert_element_type_258);  view_154 = convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_155: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_53, [-1, 9, 1]);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_262: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        amax_8: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_262, [1], True)
        sub_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_262, amax_8);  convert_element_type_262 = None
        exp_8: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_9: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
        div_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_263: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_264: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        permute_86: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_31: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_263, view_144, permute_86);  convert_element_type_263 = None
        view_157: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 384]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_87: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        clone_29: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        unsqueeze_30: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_29, -1);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_4: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_30, [0, 0, 4, 4], 0.0);  unsqueeze_30 = None
        index_4: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_4, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_4 = None
        permute_88: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
        view_159: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_88, [32, 3456, 512]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_89: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
        view_160: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_89, [32, 512, 384, 9]);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_30: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_160, memory_format = torch.contiguous_format);  view_160 = None
        view_161: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [98304, 64, 9]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_269: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        expand_25: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_161, [98304, 64, 9]);  view_161 = None
        expand_26: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_269, [98304, 9, 1]);  convert_element_type_269 = None
        bmm_12: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_25, expand_26);  expand_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_165: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [-1, 384]);  bmm_12 = None

        # No stacktrace found for following nodes
        permute_default_42: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        permute_default_43: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        permute_default_44: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_42, permute_default_43, permute_default_44, 0.1, scale = 0.125);  permute_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_173: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_7[0]

        # No stacktrace found for following nodes
        getitem_174: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_7[1]
        getitem_175: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_7[6]
        getitem_176: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_7[7];  _scaled_dot_product_flash_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_91: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3])
        clone_34: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_172: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_165, [32, -1, 6, 64]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_4: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_34, view_172], 2);  clone_34 = view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_173: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [32, 512, 768]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_277: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_278: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        view_174: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [16384, 768]);  view_173 = None
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_278, [1, 0]);  convert_element_type_278 = None
        addmm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_277, view_174, permute_92);  convert_element_type_277 = None
        view_175: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_27: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_14: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_27, 0.1);  convert_element_type_default_27 = None
        mul_64: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, view_175);  view_175 = None
        mul_65: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, add_51);  mul_65 = add_51 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_58: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_57, getitem_19);  add_57 = getitem_19 = None
        mul_66: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = None
        mul_67: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, primals_116)
        add_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, primals_117);  mul_67 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_282: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        convert_element_type_283: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_284: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16)
        view_176: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [16384, 768]);  convert_element_type_284 = None
        permute_93: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        addmm_33: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_282, view_176, permute_93);  convert_element_type_282 = None
        view_177: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_288: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_68: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, 0.5)
        mul_69: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, 0.7071067811865476);  convert_element_type_288 = None
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_60: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_70: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_60);  mul_68 = add_60 = None
        convert_element_type_289: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_290: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_291: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        view_178: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [16384, 3072]);  convert_element_type_289 = None
        permute_94: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_291, [1, 0]);  convert_element_type_291 = None
        addmm_34: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_290, view_178, permute_94);  convert_element_type_290 = None
        view_179: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 768]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_14: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_26: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_15: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_26, 0.1);  convert_element_type_default_26 = None
        mul_71: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_179);  view_179 = None
        mul_72: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_61: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_72, add_59);  mul_72 = add_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_62: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_61, getitem_21);  add_61 = getitem_21 = None
        mul_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_10);  sub_21 = None
        mul_74: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, primals_122)
        add_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, primals_123);  mul_74 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_295: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_296: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_297: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16)
        view_180: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_297, [16384, 768]);  convert_element_type_297 = None
        permute_95: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_296, [1, 0]);  convert_element_type_296 = None
        addmm_35: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_295, view_180, permute_95);  convert_element_type_295 = None
        view_181: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 384]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_301: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_302: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        permute_96: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_302, [1, 0]);  convert_element_type_302 = None
        addmm_36: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_301, view_180, permute_96);  convert_element_type_301 = None
        view_183: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 384]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_97: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_63, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_307: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_308: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_97, torch.bfloat16);  permute_97 = None
        convolution_10: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_308, convert_element_type_307, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_309: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convolution_11: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_10, convert_element_type_309, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_64: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_11, primals_130)
        convert_element_type_310: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_311: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_312: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        permute_99: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_312, [1, 0]);  convert_element_type_312 = None
        addmm_37: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_311, view_180, permute_99);  convert_element_type_311 = None
        view_185: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_186: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_187: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [32, 512, -1, 64]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_188: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_103: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_310, [0, 2, 1]);  convert_element_type_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_75: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_103, view_185);  permute_103 = view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_317: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16)
        convert_element_type_318: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        permute_104: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_318, [1, 0]);  convert_element_type_318 = None
        clone_35: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_75, memory_format = torch.contiguous_format);  mul_75 = None
        view_189: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [16384, 384]);  clone_35 = None
        mm_5: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_189, permute_104)
        view_190: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [32, 512, 54])
        add_65: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_190, convert_element_type_317);  view_190 = convert_element_type_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_191: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_65, [-1, 9, 1]);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_321: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        amax_10: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_321, [1], True)
        sub_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, amax_10);  convert_element_type_321 = None
        exp_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_11: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
        div_15: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_322: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_323: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        permute_105: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        addmm_38: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_322, view_180, permute_105);  convert_element_type_322 = None
        view_193: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_106: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        clone_36: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        unsqueeze_37: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_36, -1);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_5: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_37, [0, 0, 4, 4], 0.0);  unsqueeze_37 = None
        index_5: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_5, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_5 = None
        permute_107: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
        view_195: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_107, [32, 3456, 512]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_108: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        view_196: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_108, [32, 512, 384, 9]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_37: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_196, memory_format = torch.contiguous_format);  view_196 = None
        view_197: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [98304, 64, 9]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_328: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        expand_31: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_197, [98304, 64, 9]);  view_197 = None
        expand_32: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_328, [98304, 9, 1]);  convert_element_type_328 = None
        bmm_15: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_31, expand_32);  expand_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_201: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [-1, 384]);  bmm_15 = None

        # No stacktrace found for following nodes
        permute_default_36: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        permute_default_37: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        permute_default_38: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_36, permute_default_37, permute_default_38, 0.1, scale = 0.125);  permute_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_166: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_6[0]

        # No stacktrace found for following nodes
        getitem_167: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_6[1]
        getitem_168: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_6[6]
        getitem_169: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_6[7];  _scaled_dot_product_flash_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_110: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3])
        clone_41: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_208: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [32, -1, 6, 64]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_5: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_41, view_208], 2);  clone_41 = view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_209: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [32, 512, 768]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_336: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convert_element_type_337: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        view_210: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_209, [16384, 768]);  view_209 = None
        permute_111: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_337, [1, 0]);  convert_element_type_337 = None
        addmm_39: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_336, view_210, permute_111);  convert_element_type_336 = None
        view_211: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_25: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_17: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_25, 0.1);  convert_element_type_default_25 = None
        mul_78: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_211);  view_211 = None
        mul_79: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, add_63);  mul_79 = add_63 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_70: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_69, getitem_23);  add_69 = getitem_23 = None
        mul_80: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = None
        mul_81: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, primals_139)
        add_71: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, primals_140);  mul_81 = primals_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_341: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_342: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_343: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16)
        view_212: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_343, [16384, 768]);  convert_element_type_343 = None
        permute_112: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        addmm_40: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_341, view_212, permute_112);  convert_element_type_341 = None
        view_213: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_347: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        mul_82: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.5)
        mul_83: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.7071067811865476);  convert_element_type_347 = None
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_72: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_84: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_72);  mul_82 = add_72 = None
        convert_element_type_348: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_349: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_350: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        view_214: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [16384, 3072]);  convert_element_type_348 = None
        permute_113: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        addmm_41: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_349, view_214, permute_113);  convert_element_type_349 = None
        view_215: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_24: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_18: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_24, 0.1);  convert_element_type_default_24 = None
        mul_85: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, view_215);  view_215 = None
        mul_86: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, add_71);  mul_86 = add_71 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_74: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_73, getitem_25);  add_73 = getitem_25 = None
        mul_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_12);  sub_25 = None
        mul_88: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, primals_145)
        add_75: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, primals_146);  mul_88 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_354: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_355: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_356: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16)
        view_216: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_356, [16384, 768]);  convert_element_type_356 = None
        permute_114: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        addmm_42: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_354, view_216, permute_114);  convert_element_type_354 = None
        view_217: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 384]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_360: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_361: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        permute_115: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_43: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_360, view_216, permute_115);  convert_element_type_360 = None
        view_219: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 384]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_116: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_75, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_366: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        convert_element_type_367: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_116, torch.bfloat16);  permute_116 = None
        convolution_12: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_367, convert_element_type_366, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_368: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convolution_13: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_12, convert_element_type_368, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_76: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_13, primals_153)
        convert_element_type_369: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_370: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convert_element_type_371: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        permute_118: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_371, [1, 0]);  convert_element_type_371 = None
        addmm_44: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_370, view_216, permute_118);  convert_element_type_370 = None
        view_221: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_222: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_223: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_217, [32, 512, -1, 64]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_224: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_219, [32, 512, -1, 64]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_122: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_369, [0, 2, 1]);  convert_element_type_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_89: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_122, view_221);  permute_122 = view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_376: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16)
        convert_element_type_377: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        permute_123: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_377, [1, 0]);  convert_element_type_377 = None
        clone_42: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_89, memory_format = torch.contiguous_format);  mul_89 = None
        view_225: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [16384, 384]);  clone_42 = None
        mm_6: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_225, permute_123)
        view_226: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 512, 54])
        add_77: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_226, convert_element_type_376);  view_226 = convert_element_type_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_227: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_77, [-1, 9, 1]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_380: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        amax_12: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_380, [1], True)
        sub_26: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, amax_12);  convert_element_type_380 = None
        exp_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_13: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
        div_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_381: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_382: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        permute_124: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_382, [1, 0]);  convert_element_type_382 = None
        addmm_45: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_381, view_216, permute_124);  convert_element_type_381 = None
        view_229: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 384]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_125: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        clone_43: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        unsqueeze_44: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_43, -1);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_6: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_44, [0, 0, 4, 4], 0.0);  unsqueeze_44 = None
        index_6: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_6, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_6 = None
        permute_126: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
        view_231: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_126, [32, 3456, 512]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_127: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        view_232: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_127, [32, 512, 384, 9]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_44: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
        view_233: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [98304, 64, 9]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_387: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        expand_37: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_233, [98304, 64, 9]);  view_233 = None
        expand_38: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_387, [98304, 9, 1]);  convert_element_type_387 = None
        bmm_18: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_37, expand_38);  expand_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_237: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [-1, 384]);  bmm_18 = None

        # No stacktrace found for following nodes
        permute_default_30: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        permute_default_31: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
        permute_default_32: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_30, permute_default_31, permute_default_32, 0.1, scale = 0.125);  permute_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_159: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_5[0]

        # No stacktrace found for following nodes
        getitem_160: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_5[1]
        getitem_161: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_5[6]
        getitem_162: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_5[7];  _scaled_dot_product_flash_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_129: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3])
        clone_48: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_244: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_237, [32, -1, 6, 64]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_6: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_48, view_244], 2);  clone_48 = view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_245: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [32, 512, 768]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_395: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        convert_element_type_396: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        view_246: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [16384, 768]);  view_245 = None
        permute_130: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_396, [1, 0]);  convert_element_type_396 = None
        addmm_46: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_395, view_246, permute_130);  convert_element_type_395 = None
        view_247: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 768]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_23: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_20: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_23, 0.1);  convert_element_type_default_23 = None
        mul_92: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, view_247);  view_247 = None
        mul_93: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, add_75);  mul_93 = add_75 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_82: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_28: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_81, getitem_27);  add_81 = getitem_27 = None
        mul_94: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = None
        mul_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, primals_162)
        add_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, primals_163);  mul_95 = primals_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_400: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        convert_element_type_401: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_402: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16)
        view_248: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_402, [16384, 768]);  convert_element_type_402 = None
        permute_131: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_401, [1, 0]);  convert_element_type_401 = None
        addmm_47: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_400, view_248, permute_131);  convert_element_type_400 = None
        view_249: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_406: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        mul_96: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, 0.5)
        mul_97: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, 0.7071067811865476);  convert_element_type_406 = None
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_84: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_98: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, add_84);  mul_96 = add_84 = None
        convert_element_type_407: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_408: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        convert_element_type_409: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        view_250: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_407, [16384, 3072]);  convert_element_type_407 = None
        permute_132: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_409, [1, 0]);  convert_element_type_409 = None
        addmm_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_408, view_250, permute_132);  convert_element_type_408 = None
        view_251: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_10: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_22: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_21: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_22, 0.1);  convert_element_type_default_22 = None
        mul_99: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_251);  view_251 = None
        mul_100: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, add_83);  mul_100 = add_83 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_86: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_85, getitem_29);  add_85 = getitem_29 = None
        mul_101: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = None
        mul_102: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, primals_168)
        add_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, primals_169);  mul_102 = primals_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_413: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        convert_element_type_414: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_415: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16)
        view_252: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_415, [16384, 768]);  convert_element_type_415 = None
        permute_133: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_414, [1, 0]);  convert_element_type_414 = None
        addmm_49: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_413, view_252, permute_133);  convert_element_type_413 = None
        view_253: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 384]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_419: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        convert_element_type_420: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        permute_134: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_420, [1, 0]);  convert_element_type_420 = None
        addmm_50: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_419, view_252, permute_134);  convert_element_type_419 = None
        view_255: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 384]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_135: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_87, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_425: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        convert_element_type_426: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_135, torch.bfloat16);  permute_135 = None
        convolution_14: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_426, convert_element_type_425, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_427: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convolution_15: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_14, convert_element_type_427, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_88: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_15, primals_176)
        convert_element_type_428: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_429: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_430: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        permute_137: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_430, [1, 0]);  convert_element_type_430 = None
        addmm_51: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_429, view_252, permute_137);  convert_element_type_429 = None
        view_257: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_258: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_259: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_253, [32, 512, -1, 64]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_260: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_255, [32, 512, -1, 64]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_141: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_428, [0, 2, 1]);  convert_element_type_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_103: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_141, view_257);  permute_141 = view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_435: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16)
        convert_element_type_436: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        permute_142: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_436, [1, 0]);  convert_element_type_436 = None
        clone_49: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_103, memory_format = torch.contiguous_format);  mul_103 = None
        view_261: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [16384, 384]);  clone_49 = None
        mm_7: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_261, permute_142)
        view_262: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [32, 512, 54])
        add_89: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_262, convert_element_type_435);  view_262 = convert_element_type_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_263: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_89, [-1, 9, 1]);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_439: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        amax_14: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_439, [1], True)
        sub_30: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_439, amax_14);  convert_element_type_439 = None
        exp_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_15: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
        div_21: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_440: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_441: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        permute_143: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_441, [1, 0]);  convert_element_type_441 = None
        addmm_52: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_440, view_252, permute_143);  convert_element_type_440 = None
        view_265: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 384]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_144: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
        clone_50: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        unsqueeze_51: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_50, -1);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_7: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_51, [0, 0, 4, 4], 0.0);  unsqueeze_51 = None
        index_7: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_7, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_7 = None
        permute_145: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
        view_267: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_145, [32, 3456, 512]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_146: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        view_268: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_146, [32, 512, 384, 9]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_51: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_268, memory_format = torch.contiguous_format);  view_268 = None
        view_269: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [98304, 64, 9]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_446: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        expand_43: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_269, [98304, 64, 9]);  view_269 = None
        expand_44: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_446, [98304, 9, 1]);  convert_element_type_446 = None
        bmm_21: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_43, expand_44);  expand_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_273: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [-1, 384]);  bmm_21 = None

        # No stacktrace found for following nodes
        permute_default_24: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        permute_default_25: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        permute_default_26: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
        _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_24, permute_default_25, permute_default_26, 0.1, scale = 0.125);  permute_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_152: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_4[0]

        # No stacktrace found for following nodes
        getitem_153: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_4[1]
        getitem_154: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_4[6]
        getitem_155: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_4[7];  _scaled_dot_product_flash_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_148: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3])
        clone_55: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_280: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [32, -1, 6, 64]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_7: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_55, view_280], 2);  clone_55 = view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_281: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [32, 512, 768]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_454: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        convert_element_type_455: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        view_282: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [16384, 768]);  view_281 = None
        permute_149: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_455, [1, 0]);  convert_element_type_455 = None
        addmm_53: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_454, view_282, permute_149);  convert_element_type_454 = None
        view_283: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_21: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_23: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_21, 0.1);  convert_element_type_default_21 = None
        mul_106: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_283);  view_283 = None
        mul_107: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, 1.1111111111111112);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, add_87);  mul_107 = add_87 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_94: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_93, getitem_31);  add_93 = getitem_31 = None
        mul_108: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = None
        mul_109: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, primals_185)
        add_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, primals_186);  mul_109 = primals_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_459: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_460: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        convert_element_type_461: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16)
        view_284: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_461, [16384, 768]);  convert_element_type_461 = None
        permute_150: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_460, [1, 0]);  convert_element_type_460 = None
        addmm_54: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_459, view_284, permute_150);  convert_element_type_459 = None
        view_285: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_465: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_110: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, 0.5)
        mul_111: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, 0.7071067811865476);  convert_element_type_465 = None
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_96: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_112: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, add_96);  mul_110 = add_96 = None
        convert_element_type_466: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_467: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        convert_element_type_468: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        view_286: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_466, [16384, 3072]);  convert_element_type_466 = None
        permute_151: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_468, [1, 0]);  convert_element_type_468 = None
        addmm_55: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_467, view_286, permute_151);  convert_element_type_467 = None
        view_287: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_8: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_20: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_24: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_20, 0.1);  convert_element_type_default_20 = None
        mul_113: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, view_287);  view_287 = None
        mul_114: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_97: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, add_95);  mul_114 = add_95 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_98: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_97, getitem_33);  add_97 = getitem_33 = None
        mul_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_16);  sub_33 = None
        mul_116: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, primals_191)
        add_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, primals_192);  mul_116 = primals_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_472: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convert_element_type_473: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        convert_element_type_474: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16)
        view_288: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_474, [16384, 768]);  convert_element_type_474 = None
        permute_152: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_473, [1, 0]);  convert_element_type_473 = None
        addmm_56: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_472, view_288, permute_152);  convert_element_type_472 = None
        view_289: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 384]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_478: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        convert_element_type_479: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        permute_153: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_479, [1, 0]);  convert_element_type_479 = None
        addmm_57: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_478, view_288, permute_153);  convert_element_type_478 = None
        view_291: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 384]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_154: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_99, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_484: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        convert_element_type_485: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_154, torch.bfloat16);  permute_154 = None
        convolution_16: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_485, convert_element_type_484, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_486: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convolution_17: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_16, convert_element_type_486, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_100: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_17, primals_199)
        convert_element_type_487: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_488: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_489: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        permute_156: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_489, [1, 0]);  convert_element_type_489 = None
        addmm_58: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_488, view_288, permute_156);  convert_element_type_488 = None
        view_293: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_294: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_295: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_289, [32, 512, -1, 64]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_296: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [32, 512, -1, 64]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_160: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_487, [0, 2, 1]);  convert_element_type_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_117: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_160, view_293);  permute_160 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_494: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16)
        convert_element_type_495: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16);  primals_202 = None
        permute_161: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_495, [1, 0]);  convert_element_type_495 = None
        clone_56: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_117, memory_format = torch.contiguous_format);  mul_117 = None
        view_297: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [16384, 384]);  clone_56 = None
        mm_8: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_297, permute_161)
        view_298: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 512, 54])
        add_101: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_298, convert_element_type_494);  view_298 = convert_element_type_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_299: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_101, [-1, 9, 1]);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_498: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        amax_16: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_498, [1], True)
        sub_34: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_498, amax_16);  convert_element_type_498 = None
        exp_16: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_17: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
        div_24: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_499: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        convert_element_type_500: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        permute_162: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_500, [1, 0]);  convert_element_type_500 = None
        addmm_59: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_499, view_288, permute_162);  convert_element_type_499 = None
        view_301: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 384]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_163: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
        clone_57: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        unsqueeze_58: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_57, -1);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_8: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_58, [0, 0, 4, 4], 0.0);  unsqueeze_58 = None
        index_8: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_8, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_8 = None
        permute_164: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_8, [0, 1, 2, 4, 3, 5]);  index_8 = None
        view_303: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_164, [32, 3456, 512]);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_165: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
        view_304: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_165, [32, 512, 384, 9]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_58: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
        view_305: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [98304, 64, 9]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_505: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        expand_49: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_305, [98304, 64, 9]);  view_305 = None
        expand_50: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_505, [98304, 9, 1]);  convert_element_type_505 = None
        bmm_24: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_49, expand_50);  expand_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_309: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [-1, 384]);  bmm_24 = None

        # No stacktrace found for following nodes
        permute_default_18: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        permute_default_19: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
        permute_default_20: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_18, permute_default_19, permute_default_20, 0.1, scale = 0.125);  permute_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_145: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_3[0]

        # No stacktrace found for following nodes
        getitem_146: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_3[1]
        getitem_147: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_3[6]
        getitem_148: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_3[7];  _scaled_dot_product_flash_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_167: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3])
        clone_62: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_316: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [32, -1, 6, 64]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_8: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_62, view_316], 2);  clone_62 = view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_317: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [32, 512, 768]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_513: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        convert_element_type_514: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        view_318: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_317, [16384, 768]);  view_317 = None
        permute_168: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_514, [1, 0]);  convert_element_type_514 = None
        addmm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_513, view_318, permute_168);  convert_element_type_513 = None
        view_319: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_7: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_19: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_26: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_19, 0.1);  convert_element_type_default_19 = None
        mul_120: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, view_319);  view_319 = None
        mul_121: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_105: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, add_99);  mul_121 = add_99 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_106: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_36: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_105, getitem_35);  add_105 = getitem_35 = None
        mul_122: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = None
        mul_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, primals_208)
        add_107: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, primals_209);  mul_123 = primals_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_518: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16);  primals_211 = None
        convert_element_type_519: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16);  primals_210 = None
        convert_element_type_520: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16)
        view_320: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_520, [16384, 768]);  convert_element_type_520 = None
        permute_169: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_519, [1, 0]);  convert_element_type_519 = None
        addmm_61: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_518, view_320, permute_169);  convert_element_type_518 = None
        view_321: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_524: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        mul_124: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_524, 0.5)
        mul_125: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_524, 0.7071067811865476);  convert_element_type_524 = None
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_108: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_126: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, add_108);  mul_124 = add_108 = None
        convert_element_type_525: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_526: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        convert_element_type_527: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        view_322: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_525, [16384, 3072]);  convert_element_type_525 = None
        permute_170: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_527, [1, 0]);  convert_element_type_527 = None
        addmm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_526, view_322, permute_170);  convert_element_type_526 = None
        view_323: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_6: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_18: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_27: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_18, 0.1);  convert_element_type_default_18 = None
        mul_127: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, view_323);  view_323 = None
        mul_128: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_109: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, add_107);  mul_128 = add_107 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_110: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_109, getitem_37);  add_109 = getitem_37 = None
        mul_129: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_18);  sub_37 = None
        mul_130: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, primals_214)
        add_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, primals_215);  mul_130 = primals_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_531: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16);  primals_217 = None
        convert_element_type_532: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        convert_element_type_533: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16)
        view_324: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_533, [16384, 768]);  convert_element_type_533 = None
        permute_171: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_532, [1, 0]);  convert_element_type_532 = None
        addmm_63: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_531, view_324, permute_171);  convert_element_type_531 = None
        view_325: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 384]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_537: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16);  primals_219 = None
        convert_element_type_538: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        permute_172: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_538, [1, 0]);  convert_element_type_538 = None
        addmm_64: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_537, view_324, permute_172);  convert_element_type_537 = None
        view_327: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 384]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_173: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_111, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_543: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16);  primals_220 = None
        convert_element_type_544: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_173, torch.bfloat16);  permute_173 = None
        convolution_18: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_544, convert_element_type_543, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_545: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16);  primals_221 = None
        convolution_19: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_18, convert_element_type_545, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_112: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_19, primals_222)
        convert_element_type_546: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_547: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        convert_element_type_548: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16);  primals_223 = None
        permute_175: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_548, [1, 0]);  convert_element_type_548 = None
        addmm_65: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_547, view_324, permute_175);  convert_element_type_547 = None
        view_329: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_330: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_331: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_325, [32, 512, -1, 64]);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_332: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_327, [32, 512, -1, 64]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_179: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_546, [0, 2, 1]);  convert_element_type_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_131: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_179, view_329);  permute_179 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_553: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16)
        convert_element_type_554: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16);  primals_225 = None
        permute_180: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_554, [1, 0]);  convert_element_type_554 = None
        clone_63: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_131, memory_format = torch.contiguous_format);  mul_131 = None
        view_333: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [16384, 384]);  clone_63 = None
        mm_9: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_333, permute_180)
        view_334: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [32, 512, 54])
        add_113: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_334, convert_element_type_553);  view_334 = convert_element_type_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_335: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_113, [-1, 9, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_557: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        amax_18: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_557, [1], True)
        sub_38: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_557, amax_18);  convert_element_type_557 = None
        exp_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_19: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
        div_27: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_558: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        convert_element_type_559: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16);  primals_227 = None
        permute_181: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_559, [1, 0]);  convert_element_type_559 = None
        addmm_66: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_558, view_324, permute_181);  convert_element_type_558 = None
        view_337: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 384]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_182: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
        clone_64: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        unsqueeze_65: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_64, -1);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_9: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_65, [0, 0, 4, 4], 0.0);  unsqueeze_65 = None
        index_9: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_9, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_9 = None
        permute_183: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_9, [0, 1, 2, 4, 3, 5]);  index_9 = None
        view_339: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_183, [32, 3456, 512]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_184: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        view_340: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_184, [32, 512, 384, 9]);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_65: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_340, memory_format = torch.contiguous_format);  view_340 = None
        view_341: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [98304, 64, 9]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_564: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None
        expand_55: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_341, [98304, 64, 9]);  view_341 = None
        expand_56: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_564, [98304, 9, 1]);  convert_element_type_564 = None
        bmm_27: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_55, expand_56);  expand_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_345: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [-1, 384]);  bmm_27 = None

        # No stacktrace found for following nodes
        permute_default_12: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        permute_default_13: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
        permute_default_14: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_12, permute_default_13, permute_default_14, 0.1, scale = 0.125);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_138: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_2[0]

        # No stacktrace found for following nodes
        getitem_139: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_2[1]
        getitem_140: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_2[6]
        getitem_141: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_2[7];  _scaled_dot_product_flash_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_186: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3])
        clone_69: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_352: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [32, -1, 6, 64]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_9: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_69, view_352], 2);  clone_69 = view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_353: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [32, 512, 768]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_572: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convert_element_type_573: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        view_354: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [16384, 768]);  view_353 = None
        permute_187: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_573, [1, 0]);  convert_element_type_573 = None
        addmm_67: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_572, view_354, permute_187);  convert_element_type_572 = None
        view_355: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_17: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_29: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_17, 0.1);  convert_element_type_default_17 = None
        mul_134: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_355);  view_355 = None
        mul_135: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_117: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, add_111);  mul_135 = add_111 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_118: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_40: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_117, getitem_39);  add_117 = getitem_39 = None
        mul_136: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = None
        mul_137: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, primals_231)
        add_119: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, primals_232);  mul_137 = primals_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_577: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16);  primals_234 = None
        convert_element_type_578: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16);  primals_233 = None
        convert_element_type_579: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16)
        view_356: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_579, [16384, 768]);  convert_element_type_579 = None
        permute_188: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_578, [1, 0]);  convert_element_type_578 = None
        addmm_68: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_577, view_356, permute_188);  convert_element_type_577 = None
        view_357: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_583: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        mul_138: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, 0.5)
        mul_139: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, 0.7071067811865476);  convert_element_type_583 = None
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_120: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_140: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, add_120);  mul_138 = add_120 = None
        convert_element_type_584: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_585: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convert_element_type_586: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16);  primals_235 = None
        view_358: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_584, [16384, 3072]);  convert_element_type_584 = None
        permute_189: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_586, [1, 0]);  convert_element_type_586 = None
        addmm_69: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_585, view_358, permute_189);  convert_element_type_585 = None
        view_359: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_4: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_16: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_30: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_16, 0.1);  convert_element_type_default_16 = None
        mul_141: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, view_359);  view_359 = None
        mul_142: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_121: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, add_119);  mul_142 = add_119 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_122: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_41: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_121, getitem_41);  add_121 = getitem_41 = None
        mul_143: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_20);  sub_41 = None
        mul_144: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, primals_237)
        add_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, primals_238);  mul_144 = primals_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_590: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16);  primals_240 = None
        convert_element_type_591: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_239, torch.bfloat16);  primals_239 = None
        convert_element_type_592: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16)
        view_360: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_592, [16384, 768]);  convert_element_type_592 = None
        permute_190: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_591, [1, 0]);  convert_element_type_591 = None
        addmm_70: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_590, view_360, permute_190);  convert_element_type_590 = None
        view_361: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 384]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_596: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        convert_element_type_597: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16);  primals_241 = None
        permute_191: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_597, [1, 0]);  convert_element_type_597 = None
        addmm_71: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_596, view_360, permute_191);  convert_element_type_596 = None
        view_363: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 384]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_192: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_123, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_602: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16);  primals_243 = None
        convert_element_type_603: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_192, torch.bfloat16);  permute_192 = None
        convolution_20: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_603, convert_element_type_602, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_604: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        convolution_21: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_20, convert_element_type_604, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_124: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_21, primals_245)
        convert_element_type_605: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_606: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16);  primals_247 = None
        convert_element_type_607: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16);  primals_246 = None
        permute_194: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_607, [1, 0]);  convert_element_type_607 = None
        addmm_72: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_606, view_360, permute_194);  convert_element_type_606 = None
        view_365: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_366: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_367: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_361, [32, 512, -1, 64]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_368: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_363, [32, 512, -1, 64]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_198: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_605, [0, 2, 1]);  convert_element_type_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_145: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_198, view_365);  permute_198 = view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_612: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16)
        convert_element_type_613: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        permute_199: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_613, [1, 0]);  convert_element_type_613 = None
        clone_70: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_145, memory_format = torch.contiguous_format);  mul_145 = None
        view_369: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [16384, 384]);  clone_70 = None
        mm_10: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_369, permute_199)
        view_370: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 512, 54])
        add_125: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_370, convert_element_type_612);  view_370 = convert_element_type_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_371: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_125, [-1, 9, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_616: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        amax_20: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_616, [1], True)
        sub_42: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_616, amax_20);  convert_element_type_616 = None
        exp_20: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_21: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
        div_30: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_617: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16);  primals_251 = None
        convert_element_type_618: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16);  primals_250 = None
        permute_200: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_618, [1, 0]);  convert_element_type_618 = None
        addmm_73: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_617, view_360, permute_200);  convert_element_type_617 = None
        view_373: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [32, 512, 384]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_201: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        clone_71: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        unsqueeze_72: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_71, -1);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_10: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_72, [0, 0, 4, 4], 0.0);  unsqueeze_72 = None
        index_10: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_10, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_10 = None
        permute_202: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_10, [0, 1, 2, 4, 3, 5]);  index_10 = None
        view_375: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_202, [32, 3456, 512]);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_203: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        view_376: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_203, [32, 512, 384, 9]);  permute_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_72: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_376, memory_format = torch.contiguous_format);  view_376 = None
        view_377: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [98304, 64, 9]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_623: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None
        expand_61: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_377, [98304, 64, 9]);  view_377 = None
        expand_62: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_623, [98304, 9, 1]);  convert_element_type_623 = None
        bmm_30: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_61, expand_62);  expand_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_381: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [-1, 384]);  bmm_30 = None

        # No stacktrace found for following nodes
        permute_default_6: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        permute_default_7: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
        permute_default_8: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_6, permute_default_7, permute_default_8, 0.1, scale = 0.125);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_131: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default_1[0]

        # No stacktrace found for following nodes
        getitem_132: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_1[1]
        getitem_133: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_1[6]
        getitem_134: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_1[7];  _scaled_dot_product_flash_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_205: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3])
        clone_76: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_388: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_381, [32, -1, 6, 64]);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_10: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_76, view_388], 2);  clone_76 = view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_389: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [32, 512, 768]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_631: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        convert_element_type_632: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16);  primals_252 = None
        view_390: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [16384, 768]);  view_389 = None
        permute_206: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_632, [1, 0]);  convert_element_type_632 = None
        addmm_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_631, view_390, permute_206);  convert_element_type_631 = None
        view_391: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [32, 512, 768]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_15: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_32: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_15, 0.1);  convert_element_type_default_15 = None
        mul_148: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, view_391);  view_391 = None
        mul_149: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, 1.1111111111111112);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_129: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, add_123);  mul_149 = add_123 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_130: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_44: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_129, getitem_43);  add_129 = getitem_43 = None
        mul_150: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = None
        mul_151: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, primals_254)
        add_131: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, primals_255);  mul_151 = primals_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_636: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16);  primals_257 = None
        convert_element_type_637: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16);  primals_256 = None
        convert_element_type_638: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)
        view_392: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_638, [16384, 768]);  convert_element_type_638 = None
        permute_207: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_637, [1, 0]);  convert_element_type_637 = None
        addmm_75: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_636, view_392, permute_207);  convert_element_type_636 = None
        view_393: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_642: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_152: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, 0.5)
        mul_153: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, 0.7071067811865476);  convert_element_type_642 = None
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_132: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_154: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, add_132);  mul_152 = add_132 = None
        convert_element_type_643: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_644: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        convert_element_type_645: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16);  primals_258 = None
        view_394: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_643, [16384, 3072]);  convert_element_type_643 = None
        permute_208: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_645, [1, 0]);  convert_element_type_645 = None
        addmm_76: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_644, view_394, permute_208);  convert_element_type_644 = None
        view_395: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [32, 512, 768]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_14: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_33: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_14, 0.1);  convert_element_type_default_14 = None
        mul_155: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_395);  view_395 = None
        mul_156: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_133: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, add_131);  mul_156 = add_131 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_134: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_133, getitem_45);  add_133 = getitem_45 = None
        mul_157: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_22);  sub_45 = None
        mul_158: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, primals_260)
        add_135: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, primals_261);  mul_158 = primals_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        convert_element_type_649: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_263, torch.bfloat16);  primals_263 = None
        convert_element_type_650: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16);  primals_262 = None
        convert_element_type_651: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16)
        view_396: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_651, [16384, 768]);  convert_element_type_651 = None
        permute_209: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_650, [1, 0]);  convert_element_type_650 = None
        addmm_77: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_649, view_396, permute_209);  convert_element_type_649 = None
        view_397: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [32, 512, 384]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        convert_element_type_655: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        convert_element_type_656: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16);  primals_264 = None
        permute_210: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_656, [1, 0]);  convert_element_type_656 = None
        addmm_78: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_655, view_396, permute_210);  convert_element_type_655 = None
        view_399: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [32, 512, 384]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_211: "f32[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_135, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convert_element_type_661: "bf16[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convert_element_type_662: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.prims.convert_element_type.default(permute_211, torch.bfloat16);  permute_211 = None
        convolution_22: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_662, convert_element_type_661, None, [1], [4], [1], False, [0], 768)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convert_element_type_663: "bf16[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_267, torch.bfloat16);  primals_267 = None
        convolution_23: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_22, convert_element_type_663, None, [1], [0], [1], False, [0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_136: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_23, primals_268)
        convert_element_type_664: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        convert_element_type_665: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        convert_element_type_666: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        permute_213: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_666, [1, 0]);  convert_element_type_666 = None
        addmm_79: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_665, view_396, permute_213);  convert_element_type_665 = None
        view_401: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [32, 512, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_402: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [32, 512, -1, 64])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_403: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [32, 512, -1, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_404: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_399, [32, 512, -1, 64]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_217: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_664, [0, 2, 1]);  convert_element_type_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_159: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_217, view_401);  permute_217 = view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_671: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16)
        convert_element_type_672: "bf16[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        permute_218: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_672, [1, 0]);  convert_element_type_672 = None
        clone_77: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_159, memory_format = torch.contiguous_format);  mul_159 = None
        view_405: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [16384, 384]);  clone_77 = None
        mm_11: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_405, permute_218)
        view_406: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [32, 512, 54])
        add_137: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_406, convert_element_type_671);  view_406 = convert_element_type_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_407: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [-1, 9, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_675: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        amax_22: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_675, [1], True)
        sub_46: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_675, amax_22);  convert_element_type_675 = None
        exp_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_23: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
        div_33: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        convert_element_type_676: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16);  primals_274 = None
        convert_element_type_677: "bf16[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16);  primals_273 = None
        permute_219: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_677, [1, 0]);  convert_element_type_677 = None
        addmm_80: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_676, view_396, permute_219);  convert_element_type_676 = None
        view_409: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [32, 512, 384]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_220: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
        clone_78: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        unsqueeze_79: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_78, -1);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_11: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_79, [0, 0, 4, 4], 0.0);  unsqueeze_79 = None
        index_11: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_11, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_11 = None
        permute_221: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_11, [0, 1, 2, 4, 3, 5]);  index_11 = None
        view_411: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_221, [32, 3456, 512]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_222: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
        view_412: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_222, [32, 512, 384, 9]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_79: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_412, memory_format = torch.contiguous_format);  view_412 = None
        view_413: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [98304, 64, 9]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_682: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None
        expand_67: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_413, [98304, 64, 9]);  view_413 = None
        expand_68: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_682, [98304, 9, 1]);  convert_element_type_682 = None
        bmm_33: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_67, expand_68);  expand_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_417: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [-1, 384]);  bmm_33 = None

        # No stacktrace found for following nodes
        permute_default: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        permute_default_1: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        permute_default_2: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, 0.1, scale = 0.125);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_124: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_default[0]

        # No stacktrace found for following nodes
        getitem_125: "f32[32, 6, 512][3072, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default[1]
        getitem_126: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default[6]
        getitem_127: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default[7];  _scaled_dot_product_flash_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_224: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_124, [0, 2, 1, 3])
        clone_83: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_424: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_417, [32, -1, 6, 64]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_11: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_83, view_424], 2);  clone_83 = view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_425: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [32, 512, 768]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_690: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        convert_element_type_691: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        view_426: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [16384, 768]);  view_425 = None
        permute_225: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_691, [1, 0]);  convert_element_type_691 = None
        addmm_81: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_690, view_426, permute_225);  convert_element_type_690 = None
        view_427: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [32, 512, 768]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_13: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_35: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_13, 0.1);  convert_element_type_default_13 = None
        mul_162: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_427);  view_427 = None
        mul_163: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_141: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_163, add_135);  mul_163 = add_135 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_142: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_48: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_141, getitem_47);  add_141 = getitem_47 = None
        mul_164: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = None
        mul_165: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, primals_277)
        add_143: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, primals_278);  mul_165 = primals_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_695: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16);  primals_280 = None
        convert_element_type_696: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16);  primals_279 = None
        convert_element_type_697: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16)
        view_428: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_697, [16384, 768]);  convert_element_type_697 = None
        permute_226: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_696, [1, 0]);  convert_element_type_696 = None
        addmm_82: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_695, view_428, permute_226);  convert_element_type_695 = None
        view_429: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_701: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        mul_166: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, 0.5)
        mul_167: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, 0.7071067811865476);  convert_element_type_701 = None
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_144: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_168: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, add_144);  mul_166 = add_144 = None
        convert_element_type_702: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_703: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        convert_element_type_704: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16);  primals_281 = None
        view_430: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_702, [16384, 3072]);  convert_element_type_702 = None
        permute_227: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_704, [1, 0]);  convert_element_type_704 = None
        addmm_83: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_703, view_430, permute_227);  convert_element_type_703 = None
        view_431: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [32, 512, 768]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_12: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_36: "b8[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_12, 0.1);  convert_element_type_default_12 = None
        mul_169: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, view_431);  view_431 = None
        mul_170: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_145: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, add_143);  mul_170 = add_143 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_146: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_145, getitem_49);  add_145 = getitem_49 = None
        mul_171: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_24);  sub_49 = None
        mul_172: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, primals_283)
        add_147: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, primals_284);  mul_172 = primals_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:662 in forward, code: hidden_states = self.dense(generator_hidden_states)
        convert_element_type_708: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16);  primals_286 = None
        convert_element_type_709: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16);  primals_285 = None
        convert_element_type_710: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None
        view_432: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_710, [16384, 768]);  convert_element_type_710 = None
        permute_228: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_709, [1, 0]);  convert_element_type_709 = None
        addmm_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_708, view_432, permute_228);  convert_element_type_708 = None
        view_433: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [32, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_714: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_433, torch.float32);  view_433 = None
        mul_173: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, 0.5)
        mul_174: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, 0.7071067811865476);  convert_element_type_714 = None
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_148: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_175: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_148);  mul_173 = add_148 = None
        convert_element_type_715: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:664 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_716: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_715, torch.float32);  convert_element_type_715 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_716, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_149: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_50: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_716, getitem_51);  convert_element_type_716 = None
        mul_176: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = None
        mul_177: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, primals_287);  mul_176 = None
        add_150: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, primals_288);  mul_177 = primals_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        convert_element_type_717: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16);  primals_289 = None
        convert_element_type_718: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_719: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None
        view_434: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_719, [16384, 768]);  convert_element_type_719 = None
        permute_229: "bf16[768, 30522][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_718, [1, 0]);  convert_element_type_718 = None
        constant_pad_nd_default_2: "bf16[768, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_229, [0, 6, 0, 0])
        full_default_45: "bf16[6][1]cuda:0" = torch.ops.aten.full.default([6], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[30528][1]cuda:0" = torch.ops.aten.cat.default([convert_element_type_717, full_default_45]);  convert_element_type_717 = full_default_45 = None
        addmm_default: "bf16[16384, 30528][30528, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_434, constant_pad_nd_default_2);  cat_default = constant_pad_nd_default_2 = None
        slice_tensor: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6);  addmm_default = None
        view_435: "bf16[32, 512, 30522][15630336, 30528, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [32, 512, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:724 in forward, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_436: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [-1, 30522])
        view_437: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_290, [-1])
        convert_element_type_723: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        amax_24: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_723, [1], True)
        sub_51: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_723, amax_24);  convert_element_type_723 = None
        exp_24: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(sub_51)
        sum_25: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_52: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = None
        convert_element_type_724: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_52, torch.bfloat16);  sub_52 = None
        convert_element_type_725: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_724, torch.float32);  convert_element_type_724 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_437, -100)
        full_default_13: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_437, full_default_13);  view_437 = full_default_13 = None
        unsqueeze_86: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_725, 1, unsqueeze_86);  convert_element_type_725 = unsqueeze_86 = None
        squeeze: "f32[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_14: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_14);  neg = full_default_14 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_726: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_36: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_726);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        permute_230: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:662 in forward, code: hidden_states = self.dense(generator_hidden_states)
        permute_234: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_238: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_242: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_40: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_246: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_256: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_67, [0, 2, 1]);  expand_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_261: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_267: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_272: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_278: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_282: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_286: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_290: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_43: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_294: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_304: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_61, [0, 2, 1]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_309: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_315: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_320: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_326: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_330: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_334: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_338: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_342: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_352: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_55, [0, 2, 1]);  expand_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_357: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_363: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_368: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_374: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_378: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_48: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_382: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_386: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_49: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_390: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_400: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_49, [0, 2, 1]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_405: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_411: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_416: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_422: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_426: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_51: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_430: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_434: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_52: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_438: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_448: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_43, [0, 2, 1]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_453: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_459: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_464: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_470: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_474: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_54: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_478: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_482: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_55: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_486: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_496: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_37, [0, 2, 1]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_501: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_507: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_512: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_518: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_522: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_57: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_526: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_530: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_58: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_534: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_544: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_31, [0, 2, 1]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_549: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_555: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_560: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_566: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_570: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_60: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_574: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_578: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_61: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_582: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_592: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_25, [0, 2, 1]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_597: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_603: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_608: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_614: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_618: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_63: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_622: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_626: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_64: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_630: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_640: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_19, [0, 2, 1]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_645: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_651: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_656: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_662: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_666: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_66: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_670: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_674: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_67: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_678: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_688: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_13, [0, 2, 1]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_693: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_699: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_704: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_710: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_714: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_69: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_718: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_722: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_70: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_726: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_736: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_7, [0, 2, 1]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_741: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_747: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_752: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_758: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_762: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_72: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_766: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_770: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_73: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_774: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_784: "bf16[98304, 9, 64][576, 1, 9]cuda:0" = torch.ops.aten.permute.default(expand_1, [0, 2, 1]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_789: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_795: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_800: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_806: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_810: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_75: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_36, view_435, primals_1, primals_2, primals_4, primals_7, primals_15, primals_19, primals_24, primals_30, primals_38, primals_42, primals_47, primals_53, primals_61, primals_65, primals_70, primals_76, primals_84, primals_88, primals_93, primals_99, primals_107, primals_111, primals_116, primals_122, primals_130, primals_134, primals_139, primals_145, primals_153, primals_157, primals_162, primals_168, primals_176, primals_180, primals_185, primals_191, primals_199, primals_203, primals_208, primals_214, primals_222, primals_226, primals_231, primals_237, primals_245, primals_249, primals_254, primals_260, primals_268, primals_272, primals_277, primals_283, primals_287, primals_290, mul_1, gt, view, convert_element_type_12, convert_element_type_13, convolution, convert_element_type_14, convolution_1, addmm_2, view_9, mm, amax, sum_1, full_default_1, unsqueeze_8, permute_default_67, permute_default_68, getitem_201, getitem_202, getitem_203, getitem_204, view_30, gt_2, mul_10, view_32, addmm_5, view_34, gt_3, mul_17, view_36, convert_element_type_71, convert_element_type_72, convolution_2, convert_element_type_73, convolution_3, addmm_9, view_45, mm_1, amax_2, sum_3, permute_default_61, permute_default_62, getitem_194, getitem_195, getitem_196, getitem_197, view_66, gt_5, mul_24, view_68, addmm_12, view_70, gt_6, mul_31, view_72, convert_element_type_130, convert_element_type_131, convolution_4, convert_element_type_132, convolution_5, addmm_16, view_81, mm_2, amax_4, sum_5, permute_default_55, permute_default_56, getitem_187, getitem_188, getitem_189, getitem_190, view_102, gt_8, mul_38, view_104, addmm_19, view_106, gt_9, mul_45, view_108, convert_element_type_189, convert_element_type_190, convolution_6, convert_element_type_191, convolution_7, addmm_23, view_117, mm_3, amax_6, sum_7, permute_default_49, permute_default_50, getitem_180, getitem_181, getitem_182, getitem_183, view_138, gt_11, mul_52, view_140, addmm_26, view_142, gt_12, mul_59, view_144, convert_element_type_248, convert_element_type_249, convolution_8, convert_element_type_250, convolution_9, addmm_30, view_153, mm_4, amax_8, sum_9, permute_default_43, permute_default_44, getitem_173, getitem_174, getitem_175, getitem_176, view_174, gt_14, mul_66, view_176, addmm_33, view_178, gt_15, mul_73, view_180, convert_element_type_307, convert_element_type_308, convolution_10, convert_element_type_309, convolution_11, addmm_37, view_189, mm_5, amax_10, sum_11, permute_default_37, permute_default_38, getitem_166, getitem_167, getitem_168, getitem_169, view_210, gt_17, mul_80, view_212, addmm_40, view_214, gt_18, mul_87, view_216, convert_element_type_366, convert_element_type_367, convolution_12, convert_element_type_368, convolution_13, addmm_44, view_225, mm_6, amax_12, sum_13, permute_default_31, permute_default_32, getitem_159, getitem_160, getitem_161, getitem_162, view_246, gt_20, mul_94, view_248, addmm_47, view_250, gt_21, mul_101, view_252, convert_element_type_425, convert_element_type_426, convolution_14, convert_element_type_427, convolution_15, addmm_51, view_261, mm_7, amax_14, sum_15, permute_default_25, permute_default_26, getitem_152, getitem_153, getitem_154, getitem_155, view_282, gt_23, mul_108, view_284, addmm_54, view_286, gt_24, mul_115, view_288, convert_element_type_484, convert_element_type_485, convolution_16, convert_element_type_486, convolution_17, addmm_58, view_297, mm_8, amax_16, sum_17, permute_default_19, permute_default_20, getitem_145, getitem_146, getitem_147, getitem_148, view_318, gt_26, mul_122, view_320, addmm_61, view_322, gt_27, mul_129, view_324, convert_element_type_543, convert_element_type_544, convolution_18, convert_element_type_545, convolution_19, addmm_65, view_333, mm_9, amax_18, sum_19, permute_default_13, permute_default_14, getitem_138, getitem_139, getitem_140, getitem_141, view_354, gt_29, mul_136, view_356, addmm_68, view_358, gt_30, mul_143, view_360, convert_element_type_602, convert_element_type_603, convolution_20, convert_element_type_604, convolution_21, addmm_72, view_369, mm_10, amax_20, sum_21, permute_default_7, permute_default_8, getitem_131, getitem_132, getitem_133, getitem_134, view_390, gt_32, mul_150, view_392, addmm_75, view_394, gt_33, mul_157, view_396, convert_element_type_661, convert_element_type_662, convolution_22, convert_element_type_663, convolution_23, addmm_79, view_405, mm_11, amax_22, sum_23, permute_default_1, permute_default_2, getitem_124, getitem_125, getitem_126, getitem_127, view_426, gt_35, mul_164, view_428, addmm_82, view_430, gt_36, mul_171, view_432, addmm_84, getitem_51, rsqrt_25, view_434, view_435, amax_24, log, convert_element_type_726, permute_230, permute_234, div_39, permute_238, permute_242, div_40, permute_246, permute_256, permute_261, permute_267, permute_272, permute_278, permute_282, div_42, permute_286, permute_290, div_43, permute_294, permute_304, permute_309, permute_315, permute_320, permute_326, permute_330, div_45, permute_334, permute_338, div_46, permute_342, permute_352, permute_357, permute_363, permute_368, permute_374, permute_378, div_48, permute_382, permute_386, div_49, permute_390, permute_400, permute_405, permute_411, permute_416, permute_422, permute_426, div_51, permute_430, permute_434, div_52, permute_438, permute_448, permute_453, permute_459, permute_464, permute_470, permute_474, div_54, permute_478, permute_482, div_55, permute_486, permute_496, permute_501, permute_507, permute_512, permute_518, permute_522, div_57, permute_526, permute_530, div_58, permute_534, permute_544, permute_549, permute_555, permute_560, permute_566, permute_570, div_60, permute_574, permute_578, div_61, permute_582, permute_592, permute_597, permute_603, permute_608, permute_614, permute_618, div_63, permute_622, permute_626, div_64, permute_630, permute_640, permute_645, permute_651, permute_656, permute_662, permute_666, div_66, permute_670, permute_674, div_67, permute_678, permute_688, permute_693, permute_699, permute_704, permute_710, permute_714, div_69, permute_718, permute_722, div_70, permute_726, permute_736, permute_741, permute_747, permute_752, permute_758, permute_762, div_72, permute_766, permute_770, div_73, permute_774, permute_784, permute_789, permute_795, permute_800, permute_806, permute_810, div_75)
