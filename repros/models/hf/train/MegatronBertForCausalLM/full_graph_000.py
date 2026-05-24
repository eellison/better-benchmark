import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 512]", primals_2: "i64[16, 512]", primals_3: "f32[29056, 1024]", primals_4: "i64[1, 512]", primals_5: "f32[2, 1024]", primals_6: "f32[512, 1024]", primals_7: "f32[1024]", primals_8: "f32[1024]", primals_9: "f32[1024, 1024]", primals_10: "f32[1024]", primals_11: "f32[1024, 1024]", primals_12: "f32[1024]", primals_13: "f32[1024, 1024]", primals_14: "f32[1024]", primals_15: "f32[1024, 1024]", primals_16: "f32[1024]", primals_17: "f32[1024]", primals_18: "f32[1024]", primals_19: "f32[4096, 1024]", primals_20: "f32[4096]", primals_21: "f32[1024, 4096]", primals_22: "f32[1024]", primals_23: "f32[1024]", primals_24: "f32[1024]", primals_25: "f32[1024, 1024]", primals_26: "f32[1024]", primals_27: "f32[1024, 1024]", primals_28: "f32[1024]", primals_29: "f32[1024, 1024]", primals_30: "f32[1024]", primals_31: "f32[1024, 1024]", primals_32: "f32[1024]", primals_33: "f32[1024]", primals_34: "f32[1024]", primals_35: "f32[4096, 1024]", primals_36: "f32[4096]", primals_37: "f32[1024, 4096]", primals_38: "f32[1024]", primals_39: "f32[1024]", primals_40: "f32[1024]", primals_41: "f32[1024, 1024]", primals_42: "f32[1024]", primals_43: "f32[1024, 1024]", primals_44: "f32[1024]", primals_45: "f32[1024, 1024]", primals_46: "f32[1024]", primals_47: "f32[1024, 1024]", primals_48: "f32[1024]", primals_49: "f32[1024]", primals_50: "f32[1024]", primals_51: "f32[4096, 1024]", primals_52: "f32[4096]", primals_53: "f32[1024, 4096]", primals_54: "f32[1024]", primals_55: "f32[1024]", primals_56: "f32[1024]", primals_57: "f32[1024, 1024]", primals_58: "f32[1024]", primals_59: "f32[1024, 1024]", primals_60: "f32[1024]", primals_61: "f32[1024, 1024]", primals_62: "f32[1024]", primals_63: "f32[1024, 1024]", primals_64: "f32[1024]", primals_65: "f32[1024]", primals_66: "f32[1024]", primals_67: "f32[4096, 1024]", primals_68: "f32[4096]", primals_69: "f32[1024, 4096]", primals_70: "f32[1024]", primals_71: "f32[1024]", primals_72: "f32[1024]", primals_73: "f32[1024, 1024]", primals_74: "f32[1024]", primals_75: "f32[1024, 1024]", primals_76: "f32[1024]", primals_77: "f32[1024, 1024]", primals_78: "f32[1024]", primals_79: "f32[1024, 1024]", primals_80: "f32[1024]", primals_81: "f32[1024]", primals_82: "f32[1024]", primals_83: "f32[4096, 1024]", primals_84: "f32[4096]", primals_85: "f32[1024, 4096]", primals_86: "f32[1024]", primals_87: "f32[1024]", primals_88: "f32[1024]", primals_89: "f32[1024, 1024]", primals_90: "f32[1024]", primals_91: "f32[1024, 1024]", primals_92: "f32[1024]", primals_93: "f32[1024, 1024]", primals_94: "f32[1024]", primals_95: "f32[1024, 1024]", primals_96: "f32[1024]", primals_97: "f32[1024]", primals_98: "f32[1024]", primals_99: "f32[4096, 1024]", primals_100: "f32[4096]", primals_101: "f32[1024, 4096]", primals_102: "f32[1024]", primals_103: "f32[1024]", primals_104: "f32[1024]", primals_105: "f32[1024, 1024]", primals_106: "f32[1024]", primals_107: "f32[1024, 1024]", primals_108: "f32[1024]", primals_109: "f32[1024, 1024]", primals_110: "f32[1024]", primals_111: "f32[1024, 1024]", primals_112: "f32[1024]", primals_113: "f32[1024]", primals_114: "f32[1024]", primals_115: "f32[4096, 1024]", primals_116: "f32[4096]", primals_117: "f32[1024, 4096]", primals_118: "f32[1024]", primals_119: "f32[1024]", primals_120: "f32[1024]", primals_121: "f32[1024, 1024]", primals_122: "f32[1024]", primals_123: "f32[1024, 1024]", primals_124: "f32[1024]", primals_125: "f32[1024, 1024]", primals_126: "f32[1024]", primals_127: "f32[1024, 1024]", primals_128: "f32[1024]", primals_129: "f32[1024]", primals_130: "f32[1024]", primals_131: "f32[4096, 1024]", primals_132: "f32[4096]", primals_133: "f32[1024, 4096]", primals_134: "f32[1024]", primals_135: "f32[1024]", primals_136: "f32[1024]", primals_137: "f32[1024, 1024]", primals_138: "f32[1024]", primals_139: "f32[1024, 1024]", primals_140: "f32[1024]", primals_141: "f32[1024, 1024]", primals_142: "f32[1024]", primals_143: "f32[1024, 1024]", primals_144: "f32[1024]", primals_145: "f32[1024]", primals_146: "f32[1024]", primals_147: "f32[4096, 1024]", primals_148: "f32[4096]", primals_149: "f32[1024, 4096]", primals_150: "f32[1024]", primals_151: "f32[1024]", primals_152: "f32[1024]", primals_153: "f32[1024, 1024]", primals_154: "f32[1024]", primals_155: "f32[1024, 1024]", primals_156: "f32[1024]", primals_157: "f32[1024, 1024]", primals_158: "f32[1024]", primals_159: "f32[1024, 1024]", primals_160: "f32[1024]", primals_161: "f32[1024]", primals_162: "f32[1024]", primals_163: "f32[4096, 1024]", primals_164: "f32[4096]", primals_165: "f32[1024, 4096]", primals_166: "f32[1024]", primals_167: "f32[1024]", primals_168: "f32[1024]", primals_169: "f32[1024, 1024]", primals_170: "f32[1024]", primals_171: "f32[1024, 1024]", primals_172: "f32[1024]", primals_173: "f32[1024, 1024]", primals_174: "f32[1024]", primals_175: "f32[1024, 1024]", primals_176: "f32[1024]", primals_177: "f32[1024]", primals_178: "f32[1024]", primals_179: "f32[4096, 1024]", primals_180: "f32[4096]", primals_181: "f32[1024, 4096]", primals_182: "f32[1024]", primals_183: "f32[1024]", primals_184: "f32[1024]", primals_185: "f32[1024, 1024]", primals_186: "f32[1024]", primals_187: "f32[1024, 1024]", primals_188: "f32[1024]", primals_189: "f32[1024, 1024]", primals_190: "f32[1024]", primals_191: "f32[1024, 1024]", primals_192: "f32[1024]", primals_193: "f32[1024]", primals_194: "f32[1024]", primals_195: "f32[4096, 1024]", primals_196: "f32[4096]", primals_197: "f32[1024, 4096]", primals_198: "f32[1024]", primals_199: "f32[1024]", primals_200: "f32[1024]", primals_201: "f32[1024, 1024]", primals_202: "f32[1024]", primals_203: "f32[1024, 1024]", primals_204: "f32[1024]", primals_205: "f32[1024, 1024]", primals_206: "f32[1024]", primals_207: "f32[1024, 1024]", primals_208: "f32[1024]", primals_209: "f32[1024]", primals_210: "f32[1024]", primals_211: "f32[4096, 1024]", primals_212: "f32[4096]", primals_213: "f32[1024, 4096]", primals_214: "f32[1024]", primals_215: "f32[1024]", primals_216: "f32[1024]", primals_217: "f32[1024, 1024]", primals_218: "f32[1024]", primals_219: "f32[1024, 1024]", primals_220: "f32[1024]", primals_221: "f32[1024, 1024]", primals_222: "f32[1024]", primals_223: "f32[1024, 1024]", primals_224: "f32[1024]", primals_225: "f32[1024]", primals_226: "f32[1024]", primals_227: "f32[4096, 1024]", primals_228: "f32[4096]", primals_229: "f32[1024, 4096]", primals_230: "f32[1024]", primals_231: "f32[1024]", primals_232: "f32[1024]", primals_233: "f32[1024, 1024]", primals_234: "f32[1024]", primals_235: "f32[1024, 1024]", primals_236: "f32[1024]", primals_237: "f32[1024, 1024]", primals_238: "f32[1024]", primals_239: "f32[1024, 1024]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_242: "f32[1024]", primals_243: "f32[4096, 1024]", primals_244: "f32[4096]", primals_245: "f32[1024, 4096]", primals_246: "f32[1024]", primals_247: "f32[1024]", primals_248: "f32[1024]", primals_249: "f32[1024, 1024]", primals_250: "f32[1024]", primals_251: "f32[1024, 1024]", primals_252: "f32[1024]", primals_253: "f32[1024, 1024]", primals_254: "f32[1024]", primals_255: "f32[1024, 1024]", primals_256: "f32[1024]", primals_257: "f32[1024]", primals_258: "f32[1024]", primals_259: "f32[4096, 1024]", primals_260: "f32[4096]", primals_261: "f32[1024, 4096]", primals_262: "f32[1024]", primals_263: "f32[1024]", primals_264: "f32[1024]", primals_265: "f32[1024, 1024]", primals_266: "f32[1024]", primals_267: "f32[1024, 1024]", primals_268: "f32[1024]", primals_269: "f32[1024, 1024]", primals_270: "f32[1024]", primals_271: "f32[1024, 1024]", primals_272: "f32[1024]", primals_273: "f32[1024]", primals_274: "f32[1024]", primals_275: "f32[4096, 1024]", primals_276: "f32[4096]", primals_277: "f32[1024, 4096]", primals_278: "f32[1024]", primals_279: "f32[1024]", primals_280: "f32[1024]", primals_281: "f32[1024, 1024]", primals_282: "f32[1024]", primals_283: "f32[1024, 1024]", primals_284: "f32[1024]", primals_285: "f32[1024, 1024]", primals_286: "f32[1024]", primals_287: "f32[1024, 1024]", primals_288: "f32[1024]", primals_289: "f32[1024]", primals_290: "f32[1024]", primals_291: "f32[4096, 1024]", primals_292: "f32[4096]", primals_293: "f32[1024, 4096]", primals_294: "f32[1024]", primals_295: "f32[1024]", primals_296: "f32[1024]", primals_297: "f32[1024, 1024]", primals_298: "f32[1024]", primals_299: "f32[1024, 1024]", primals_300: "f32[1024]", primals_301: "f32[1024, 1024]", primals_302: "f32[1024]", primals_303: "f32[1024, 1024]", primals_304: "f32[1024]", primals_305: "f32[1024]", primals_306: "f32[1024]", primals_307: "f32[4096, 1024]", primals_308: "f32[4096]", primals_309: "f32[1024, 4096]", primals_310: "f32[1024]", primals_311: "f32[1024]", primals_312: "f32[1024]", primals_313: "f32[1024, 1024]", primals_314: "f32[1024]", primals_315: "f32[1024, 1024]", primals_316: "f32[1024]", primals_317: "f32[1024, 1024]", primals_318: "f32[1024]", primals_319: "f32[1024, 1024]", primals_320: "f32[1024]", primals_321: "f32[1024]", primals_322: "f32[1024]", primals_323: "f32[4096, 1024]", primals_324: "f32[4096]", primals_325: "f32[1024, 4096]", primals_326: "f32[1024]", primals_327: "f32[1024]", primals_328: "f32[1024]", primals_329: "f32[1024, 1024]", primals_330: "f32[1024]", primals_331: "f32[1024, 1024]", primals_332: "f32[1024]", primals_333: "f32[1024, 1024]", primals_334: "f32[1024]", primals_335: "f32[1024, 1024]", primals_336: "f32[1024]", primals_337: "f32[1024]", primals_338: "f32[1024]", primals_339: "f32[4096, 1024]", primals_340: "f32[4096]", primals_341: "f32[1024, 4096]", primals_342: "f32[1024]", primals_343: "f32[1024]", primals_344: "f32[1024]", primals_345: "f32[1024, 1024]", primals_346: "f32[1024]", primals_347: "f32[1024, 1024]", primals_348: "f32[1024]", primals_349: "f32[1024, 1024]", primals_350: "f32[1024]", primals_351: "f32[1024, 1024]", primals_352: "f32[1024]", primals_353: "f32[1024]", primals_354: "f32[1024]", primals_355: "f32[4096, 1024]", primals_356: "f32[4096]", primals_357: "f32[1024, 4096]", primals_358: "f32[1024]", primals_359: "f32[1024]", primals_360: "f32[1024]", primals_361: "f32[1024, 1024]", primals_362: "f32[1024]", primals_363: "f32[1024, 1024]", primals_364: "f32[1024]", primals_365: "f32[1024, 1024]", primals_366: "f32[1024]", primals_367: "f32[1024, 1024]", primals_368: "f32[1024]", primals_369: "f32[1024]", primals_370: "f32[1024]", primals_371: "f32[4096, 1024]", primals_372: "f32[4096]", primals_373: "f32[1024, 4096]", primals_374: "f32[1024]", primals_375: "f32[1024]", primals_376: "f32[1024]", primals_377: "f32[1024, 1024]", primals_378: "f32[1024]", primals_379: "f32[1024, 1024]", primals_380: "f32[1024]", primals_381: "f32[1024, 1024]", primals_382: "f32[1024]", primals_383: "f32[1024, 1024]", primals_384: "f32[1024]", primals_385: "f32[1024]", primals_386: "f32[1024]", primals_387: "f32[4096, 1024]", primals_388: "f32[4096]", primals_389: "f32[1024, 4096]", primals_390: "f32[1024]", primals_391: "f32[1024]", primals_392: "f32[1024]", primals_393: "f32[1024, 1024]", primals_394: "f32[1024]", primals_395: "f32[1024]", primals_396: "f32[1024]", primals_397: "f32[29056]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:628 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[16, 512]" = torch.ops.aten.full.default([16, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(primals_3, primals_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(primals_5, full_default);  primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:91 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_6, primals_4);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[49]" = torch.ops.prims.inductor_seeds.default(49, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:98 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_48: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(mul_2, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  getitem_1 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_7)
        add_3: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_4, primals_8);  mul_4 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_3, [8192, 1024]);  add_3 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_10, view, permute);  primals_10 = permute = None
        view_1: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm, [16, 512, 1024]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_2: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1, [16, 512, -1, 64]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_1: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_12, view, permute_2);  primals_12 = permute_2 = None
        view_4: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_1, [16, 512, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_5: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_4, [16, 512, -1, 64]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_2: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_14, view, permute_4);  primals_14 = permute_4 = None
        view_7: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_2, [16, 512, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_8: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_7, [16, 512, -1, 64]);  view_7 = None

        # No stacktrace found for following nodes
        permute_default_138: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_default_139: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_default_140: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_138, permute_default_139, permute_default_140, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_261: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_23[0]

        # No stacktrace found for following nodes
        getitem_262: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_23[1]
        getitem_263: "i64[]" = _scaled_dot_product_efficient_attention_default_23[2]
        getitem_264: "i64[]" = _scaled_dot_product_efficient_attention_default_23[3];  _scaled_dot_product_efficient_attention_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_7: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3])
        clone_3: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_15: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_3, [16, 512, 1024]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_15, [8192, 1024]);  view_15 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_3: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_16, view_16, permute_8);  primals_16 = permute_8 = None
        view_17: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_3, [16, 512, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_47: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_2: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_47, 0.1);  inductor_random_default_47 = None
        mul_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_8: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_5: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_2, mul_8);  mul_2 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[16, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        mul_9: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_10: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_9, primals_17)
        add_7: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_10, primals_18);  mul_10 = primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_7, [8192, 1024]);  add_7 = None
        permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        addmm_4: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_20, view_18, permute_9);  primals_20 = permute_9 = None
        view_19: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_4, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_11: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_12: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_13, [8192, 4096]);  mul_13 = None
        permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_5: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_22, view_20, permute_10);  primals_22 = permute_10 = None
        view_21: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_5, [16, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_46: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_3: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_46, 0.1);  inductor_random_default_46 = None
        mul_14: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_15: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_9: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_5, mul_15);  add_5 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[16, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        mul_16: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_17: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_16, primals_23)
        add_11: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_17, primals_24);  mul_17 = primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_22: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_11, [8192, 1024]);  add_11 = None
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_6: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_26, view_22, permute_11);  primals_26 = permute_11 = None
        view_23: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_6, [16, 512, 1024]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_24: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_23, [16, 512, -1, 64]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_13: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_7: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_28, view_22, permute_13);  primals_28 = permute_13 = None
        view_26: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_7, [16, 512, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_27: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_26, [16, 512, -1, 64]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_15: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_8: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_30, view_22, permute_15);  primals_30 = permute_15 = None
        view_29: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_8, [16, 512, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_30: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_29, [16, 512, -1, 64]);  view_29 = None

        # No stacktrace found for following nodes
        permute_default_132: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        permute_default_133: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        permute_default_134: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_132, permute_default_133, permute_default_134, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_254: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_22[0]

        # No stacktrace found for following nodes
        getitem_255: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_22[1]
        getitem_256: "i64[]" = _scaled_dot_product_efficient_attention_default_22[2]
        getitem_257: "i64[]" = _scaled_dot_product_efficient_attention_default_22[3];  _scaled_dot_product_efficient_attention_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_18: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_254, [0, 2, 1, 3])
        clone_7: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_37: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_7, [16, 512, 1024]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_37, [8192, 1024]);  view_37 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_31, [1, 0])
        addmm_9: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_32, view_38, permute_19);  primals_32 = permute_19 = None
        view_39: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_9, [16, 512, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_45: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_5: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_20: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_21: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_13: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_9, mul_21);  add_9 = mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[16, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        mul_22: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_23: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, primals_33)
        add_15: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_23, primals_34);  mul_23 = primals_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_15, [8192, 1024]);  add_15 = None
        permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_10: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_36, view_40, permute_20);  primals_36 = permute_20 = None
        view_41: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_10, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_24: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_25: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_26: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_24, add_16);  mul_24 = add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_26, [8192, 4096]);  mul_26 = None
        permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_37, [1, 0])
        addmm_11: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_38, view_42, permute_21);  primals_38 = permute_21 = None
        view_43: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_11, [16, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_44: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_6: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_27: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_28: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_27, 1.1111111111111112);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_17: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_13, mul_28);  add_13 = mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[16, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        mul_29: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
        mul_30: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, primals_39)
        add_19: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_30, primals_40);  mul_30 = primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_44: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_19, [8192, 1024]);  add_19 = None
        permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_12: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_42, view_44, permute_22);  primals_42 = permute_22 = None
        view_45: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_12, [16, 512, 1024]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_46: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_45, [16, 512, -1, 64]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_24: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        addmm_13: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_44, view_44, permute_24);  primals_44 = permute_24 = None
        view_48: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_13, [16, 512, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_49: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_48, [16, 512, -1, 64]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_26: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_14: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_46, view_44, permute_26);  primals_46 = permute_26 = None
        view_51: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_14, [16, 512, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_52: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_51, [16, 512, -1, 64]);  view_51 = None

        # No stacktrace found for following nodes
        permute_default_126: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        permute_default_127: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        permute_default_128: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_126, permute_default_127, permute_default_128, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_247: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_21[0]

        # No stacktrace found for following nodes
        getitem_248: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_21[1]
        getitem_249: "i64[]" = _scaled_dot_product_efficient_attention_default_21[2]
        getitem_250: "i64[]" = _scaled_dot_product_efficient_attention_default_21[3];  _scaled_dot_product_efficient_attention_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_29: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_247, [0, 2, 1, 3])
        clone_11: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_59: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_11, [16, 512, 1024]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_59, [8192, 1024]);  view_59 = None
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_47, [1, 0])
        addmm_15: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_48, view_60, permute_30);  primals_48 = permute_30 = None
        view_61: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_15, [16, 512, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_43: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_8: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_43, 0.1);  inductor_random_default_43 = None
        mul_33: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_34: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, 1.1111111111111112);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_21: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_17, mul_34);  add_17 = mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10: "f32[16, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[16, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_22: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  getitem_11 = None
        mul_35: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
        mul_36: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_35, primals_49)
        add_23: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_36, primals_50);  mul_36 = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_23, [8192, 1024]);  add_23 = None
        permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_16: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_52, view_62, permute_31);  primals_52 = permute_31 = None
        view_63: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_16, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_37: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_38: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_39: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_37, add_24);  mul_37 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_39, [8192, 4096]);  mul_39 = None
        permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_53, [1, 0])
        addmm_17: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_54, view_64, permute_32);  primals_54 = permute_32 = None
        view_65: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_17, [16, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_42: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_9: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_42, 0.1);  inductor_random_default_42 = None
        mul_40: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_41: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_25: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_21, mul_41);  add_21 = mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[16, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[16, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  getitem_13 = None
        mul_42: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
        mul_43: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_42, primals_55)
        add_27: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_43, primals_56);  mul_43 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_66: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_27, [8192, 1024]);  add_27 = None
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_18: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_58, view_66, permute_33);  primals_58 = permute_33 = None
        view_67: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_18, [16, 512, 1024]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_68: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_67, [16, 512, -1, 64]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_35: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_19: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_60, view_66, permute_35);  primals_60 = permute_35 = None
        view_70: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_19, [16, 512, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_71: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_70, [16, 512, -1, 64]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_37: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        addmm_20: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_62, view_66, permute_37);  primals_62 = permute_37 = None
        view_73: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_20, [16, 512, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_74: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_73, [16, 512, -1, 64]);  view_73 = None

        # No stacktrace found for following nodes
        permute_default_120: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        permute_default_121: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_default_122: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_120, permute_default_121, permute_default_122, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_240: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_20[0]

        # No stacktrace found for following nodes
        getitem_241: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_20[1]
        getitem_242: "i64[]" = _scaled_dot_product_efficient_attention_default_20[2]
        getitem_243: "i64[]" = _scaled_dot_product_efficient_attention_default_20[3];  _scaled_dot_product_efficient_attention_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_40: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3])
        clone_15: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_81: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_15, [16, 512, 1024]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_81, [8192, 1024]);  view_81 = None
        permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        addmm_21: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_64, view_82, permute_41);  primals_64 = permute_41 = None
        view_83: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_21, [16, 512, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_41: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_11: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_41, 0.1);  inductor_random_default_41 = None
        mul_46: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_47: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_46, 1.1111111111111112);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_29: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_25, mul_47);  add_25 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14: "f32[16, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[16, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_30: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  getitem_15 = None
        mul_48: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
        mul_49: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_48, primals_65)
        add_31: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_49, primals_66);  mul_49 = primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_31, [8192, 1024]);  add_31 = None
        permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        addmm_22: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_68, view_84, permute_42);  primals_68 = permute_42 = None
        view_85: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_22, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_50: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_51: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_52: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_50, add_32);  mul_50 = add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_52, [8192, 4096]);  mul_52 = None
        permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_69, [1, 0])
        addmm_23: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_70, view_86, permute_43);  primals_70 = permute_43 = None
        view_87: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_23, [16, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_40: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_12: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_53: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_54: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_33: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_29, mul_54);  add_29 = mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[16, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[16, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  getitem_17 = None
        mul_55: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
        mul_56: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_55, primals_71)
        add_35: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_56, primals_72);  mul_56 = primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_88: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_35, [8192, 1024]);  add_35 = None
        permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_73, [1, 0])
        addmm_24: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_74, view_88, permute_44);  primals_74 = permute_44 = None
        view_89: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_24, [16, 512, 1024]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_90: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_89, [16, 512, -1, 64]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_46: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_25: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_76, view_88, permute_46);  primals_76 = permute_46 = None
        view_92: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_25, [16, 512, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_93: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_92, [16, 512, -1, 64]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_48: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_26: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_78, view_88, permute_48);  primals_78 = permute_48 = None
        view_95: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_26, [16, 512, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_96: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_95, [16, 512, -1, 64]);  view_95 = None

        # No stacktrace found for following nodes
        permute_default_114: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        permute_default_115: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        permute_default_116: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_114, permute_default_115, permute_default_116, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_233: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_19[0]

        # No stacktrace found for following nodes
        getitem_234: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_19[1]
        getitem_235: "i64[]" = _scaled_dot_product_efficient_attention_default_19[2]
        getitem_236: "i64[]" = _scaled_dot_product_efficient_attention_default_19[3];  _scaled_dot_product_efficient_attention_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_51: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_233, [0, 2, 1, 3])
        clone_19: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_103: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_19, [16, 512, 1024]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_103, [8192, 1024]);  view_103 = None
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_79, [1, 0])
        addmm_27: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_80, view_104, permute_52);  primals_80 = permute_52 = None
        view_105: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_27, [16, 512, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_39: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_14: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_59: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_60: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_59, 1.1111111111111112);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_37: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_33, mul_60);  add_33 = mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18: "f32[16, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[16, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_38: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  getitem_19 = None
        mul_61: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_62: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_61, primals_81)
        add_39: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_62, primals_82);  mul_62 = primals_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_39, [8192, 1024]);  add_39 = None
        permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_28: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_84, view_106, permute_53);  primals_84 = permute_53 = None
        view_107: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_28, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_63: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_64: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_65: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_63, add_40);  mul_63 = add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_65, [8192, 4096]);  mul_65 = None
        permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_85, [1, 0])
        addmm_29: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_86, view_108, permute_54);  primals_86 = permute_54 = None
        view_109: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_29, [16, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_38: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_15: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_66: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_67: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_41: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_37, mul_67);  add_37 = mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[16, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[16, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  getitem_21 = None
        mul_68: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_69: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_68, primals_87)
        add_43: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_69, primals_88);  mul_69 = primals_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_110: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_43, [8192, 1024]);  add_43 = None
        permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_30: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_90, view_110, permute_55);  primals_90 = permute_55 = None
        view_111: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_30, [16, 512, 1024]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_112: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_111, [16, 512, -1, 64]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_57: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_31: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_92, view_110, permute_57);  primals_92 = permute_57 = None
        view_114: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_31, [16, 512, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_115: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_114, [16, 512, -1, 64]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_59: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_32: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_94, view_110, permute_59);  primals_94 = permute_59 = None
        view_117: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_32, [16, 512, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_118: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_117, [16, 512, -1, 64]);  view_117 = None

        # No stacktrace found for following nodes
        permute_default_108: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        permute_default_109: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        permute_default_110: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_108, permute_default_109, permute_default_110, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_226: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_18[0]

        # No stacktrace found for following nodes
        getitem_227: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_18[1]
        getitem_228: "i64[]" = _scaled_dot_product_efficient_attention_default_18[2]
        getitem_229: "i64[]" = _scaled_dot_product_efficient_attention_default_18[3];  _scaled_dot_product_efficient_attention_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_62: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_226, [0, 2, 1, 3])
        clone_23: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_125: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_23, [16, 512, 1024]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_125, [8192, 1024]);  view_125 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_95, [1, 0])
        addmm_33: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_96, view_126, permute_63);  primals_96 = permute_63 = None
        view_127: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_33, [16, 512, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_37: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_17: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_72: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_73: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_45: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_41, mul_73);  add_41 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22: "f32[16, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[16, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_46: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  getitem_23 = None
        mul_74: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
        mul_75: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_74, primals_97)
        add_47: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_75, primals_98);  mul_75 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_47, [8192, 1024]);  add_47 = None
        permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_34: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_100, view_128, permute_64);  primals_100 = permute_64 = None
        view_129: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_34, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_76: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_77: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_78: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_76, add_48);  mul_76 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_78, [8192, 4096]);  mul_78 = None
        permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_101, [1, 0])
        addmm_35: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_102, view_130, permute_65);  primals_102 = permute_65 = None
        view_131: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_35, [16, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_36: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_18: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_79: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_80: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_49: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_45, mul_80);  add_45 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[16, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[16, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  getitem_25 = None
        mul_81: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
        mul_82: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_81, primals_103)
        add_51: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_82, primals_104);  mul_82 = primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_132: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_51, [8192, 1024]);  add_51 = None
        permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        addmm_36: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_106, view_132, permute_66);  primals_106 = permute_66 = None
        view_133: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_36, [16, 512, 1024]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_134: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_133, [16, 512, -1, 64]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_68: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_107, [1, 0])
        addmm_37: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_108, view_132, permute_68);  primals_108 = permute_68 = None
        view_136: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_37, [16, 512, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_137: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_136, [16, 512, -1, 64]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        addmm_38: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_110, view_132, permute_70);  primals_110 = permute_70 = None
        view_139: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_38, [16, 512, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_140: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_139, [16, 512, -1, 64]);  view_139 = None

        # No stacktrace found for following nodes
        permute_default_102: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        permute_default_103: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        permute_default_104: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_102, permute_default_103, permute_default_104, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_219: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_17[0]

        # No stacktrace found for following nodes
        getitem_220: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_17[1]
        getitem_221: "i64[]" = _scaled_dot_product_efficient_attention_default_17[2]
        getitem_222: "i64[]" = _scaled_dot_product_efficient_attention_default_17[3];  _scaled_dot_product_efficient_attention_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_73: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_219, [0, 2, 1, 3])
        clone_27: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_147: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_27, [16, 512, 1024]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_147, [8192, 1024]);  view_147 = None
        permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_111, [1, 0])
        addmm_39: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_112, view_148, permute_74);  primals_112 = permute_74 = None
        view_149: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_39, [16, 512, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_35: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_20: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_85: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_86: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_53: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_49, mul_86);  add_49 = mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26: "f32[16, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[16, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_54: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  getitem_27 = None
        mul_87: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
        mul_88: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, primals_113)
        add_55: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_88, primals_114);  mul_88 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_55, [8192, 1024]);  add_55 = None
        permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_115, [1, 0])
        addmm_40: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_116, view_150, permute_75);  primals_116 = permute_75 = None
        view_151: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_40, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_89: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_90: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_91: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_89, add_56);  mul_89 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_91, [8192, 4096]);  mul_91 = None
        permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_117, [1, 0])
        addmm_41: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_118, view_152, permute_76);  primals_118 = permute_76 = None
        view_153: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_41, [16, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_34: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_21: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_92: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_93: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_57: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_53, mul_93);  add_53 = mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[16, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[16, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  getitem_29 = None
        mul_94: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
        mul_95: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, primals_119)
        add_59: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_95, primals_120);  mul_95 = primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_154: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_59, [8192, 1024]);  add_59 = None
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_121, [1, 0])
        addmm_42: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_122, view_154, permute_77);  primals_122 = permute_77 = None
        view_155: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_42, [16, 512, 1024]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_156: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_155, [16, 512, -1, 64]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_79: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_123, [1, 0])
        addmm_43: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_124, view_154, permute_79);  primals_124 = permute_79 = None
        view_158: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_43, [16, 512, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_159: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_158, [16, 512, -1, 64]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_81: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_125, [1, 0])
        addmm_44: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_126, view_154, permute_81);  primals_126 = permute_81 = None
        view_161: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_44, [16, 512, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_162: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_161, [16, 512, -1, 64]);  view_161 = None

        # No stacktrace found for following nodes
        permute_default_96: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        permute_default_97: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        permute_default_98: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_96, permute_default_97, permute_default_98, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_212: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_16[0]

        # No stacktrace found for following nodes
        getitem_213: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_16[1]
        getitem_214: "i64[]" = _scaled_dot_product_efficient_attention_default_16[2]
        getitem_215: "i64[]" = _scaled_dot_product_efficient_attention_default_16[3];  _scaled_dot_product_efficient_attention_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_84: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3])
        clone_31: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_169: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_31, [16, 512, 1024]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_169, [8192, 1024]);  view_169 = None
        permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_127, [1, 0])
        addmm_45: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_128, view_170, permute_85);  primals_128 = permute_85 = None
        view_171: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_45, [16, 512, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_33: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_23: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_98: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_99: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_61: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_57, mul_99);  add_57 = mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30: "f32[16, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[16, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_62: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  getitem_31 = None
        mul_100: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
        mul_101: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_100, primals_129)
        add_63: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_101, primals_130);  mul_101 = primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_63, [8192, 1024]);  add_63 = None
        permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_131, [1, 0])
        addmm_46: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_132, view_172, permute_86);  primals_132 = permute_86 = None
        view_173: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_46, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_102: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_103: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_104: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_102, add_64);  mul_102 = add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_104, [8192, 4096]);  mul_104 = None
        permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_133, [1, 0])
        addmm_47: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_134, view_174, permute_87);  primals_134 = permute_87 = None
        view_175: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_47, [16, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_32: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_24: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_105: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_106: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_65: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_61, mul_106);  add_61 = mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[16, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[16, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  getitem_33 = None
        mul_107: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
        mul_108: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_107, primals_135)
        add_67: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_108, primals_136);  mul_108 = primals_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_176: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_67, [8192, 1024]);  add_67 = None
        permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_137, [1, 0])
        addmm_48: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_138, view_176, permute_88);  primals_138 = permute_88 = None
        view_177: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_48, [16, 512, 1024]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_178: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_177, [16, 512, -1, 64]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_90: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_139, [1, 0])
        addmm_49: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_140, view_176, permute_90);  primals_140 = permute_90 = None
        view_180: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_49, [16, 512, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_181: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_180, [16, 512, -1, 64]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_92: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_141, [1, 0])
        addmm_50: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_142, view_176, permute_92);  primals_142 = permute_92 = None
        view_183: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_50, [16, 512, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_184: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_183, [16, 512, -1, 64]);  view_183 = None

        # No stacktrace found for following nodes
        permute_default_90: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        permute_default_91: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        permute_default_92: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_90, permute_default_91, permute_default_92, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_205: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_15[0]

        # No stacktrace found for following nodes
        getitem_206: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_15[1]
        getitem_207: "i64[]" = _scaled_dot_product_efficient_attention_default_15[2]
        getitem_208: "i64[]" = _scaled_dot_product_efficient_attention_default_15[3];  _scaled_dot_product_efficient_attention_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_95: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3])
        clone_35: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_191: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_35, [16, 512, 1024]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_191, [8192, 1024]);  view_191 = None
        permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0])
        addmm_51: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_144, view_192, permute_96);  primals_144 = permute_96 = None
        view_193: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_51, [16, 512, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_31: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_26: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_111: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_112: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_69: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_65, mul_112);  add_65 = mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34: "f32[16, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[16, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_70: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  getitem_35 = None
        mul_113: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
        mul_114: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, primals_145)
        add_71: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_114, primals_146);  mul_114 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_71, [8192, 1024]);  add_71 = None
        permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_147, [1, 0])
        addmm_52: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_148, view_194, permute_97);  primals_148 = permute_97 = None
        view_195: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_52, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_115: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_116: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_117: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_115, add_72);  mul_115 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_117, [8192, 4096]);  mul_117 = None
        permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_149, [1, 0])
        addmm_53: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_150, view_196, permute_98);  primals_150 = permute_98 = None
        view_197: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_53, [16, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_30: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_27: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_118: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_119: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_73: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_69, mul_119);  add_69 = mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[16, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[16, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  getitem_37 = None
        mul_120: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
        mul_121: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, primals_151)
        add_75: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_121, primals_152);  mul_121 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_198: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_75, [8192, 1024]);  add_75 = None
        permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_153, [1, 0])
        addmm_54: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_154, view_198, permute_99);  primals_154 = permute_99 = None
        view_199: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_54, [16, 512, 1024]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_200: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_199, [16, 512, -1, 64]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_101: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0])
        addmm_55: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_156, view_198, permute_101);  primals_156 = permute_101 = None
        view_202: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_55, [16, 512, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_203: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_202, [16, 512, -1, 64]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_103: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_157, [1, 0])
        addmm_56: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_158, view_198, permute_103);  primals_158 = permute_103 = None
        view_205: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_56, [16, 512, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_206: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_205, [16, 512, -1, 64]);  view_205 = None

        # No stacktrace found for following nodes
        permute_default_84: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        permute_default_85: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        permute_default_86: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_84, permute_default_85, permute_default_86, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_198: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_14[0]

        # No stacktrace found for following nodes
        getitem_199: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_14[1]
        getitem_200: "i64[]" = _scaled_dot_product_efficient_attention_default_14[2]
        getitem_201: "i64[]" = _scaled_dot_product_efficient_attention_default_14[3];  _scaled_dot_product_efficient_attention_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_106: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3])
        clone_39: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_213: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_39, [16, 512, 1024]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_213, [8192, 1024]);  view_213 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_159, [1, 0])
        addmm_57: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_160, view_214, permute_107);  primals_160 = permute_107 = None
        view_215: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_57, [16, 512, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_29: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_29: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_124: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_125: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_77: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_73, mul_125);  add_73 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[16, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_78: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  getitem_39 = None
        mul_126: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
        mul_127: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_126, primals_161)
        add_79: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_127, primals_162);  mul_127 = primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_79, [8192, 1024]);  add_79 = None
        permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_163, [1, 0])
        addmm_58: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_164, view_216, permute_108);  primals_164 = permute_108 = None
        view_217: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_58, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_128: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_129: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_130: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_128, add_80);  mul_128 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_130, [8192, 4096]);  mul_130 = None
        permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_165, [1, 0])
        addmm_59: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_166, view_218, permute_109);  primals_166 = permute_109 = None
        view_219: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_59, [16, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_28: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_30: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_131: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_132: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_131, 1.1111111111111112);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_81: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_77, mul_132);  add_77 = mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[16, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  getitem_41 = None
        mul_133: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
        mul_134: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_133, primals_167)
        add_83: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_134, primals_168);  mul_134 = primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_220: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_83, [8192, 1024]);  add_83 = None
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_169, [1, 0])
        addmm_60: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_170, view_220, permute_110);  primals_170 = permute_110 = None
        view_221: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_60, [16, 512, 1024]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_222: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_221, [16, 512, -1, 64]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_112: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_171, [1, 0])
        addmm_61: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_172, view_220, permute_112);  primals_172 = permute_112 = None
        view_224: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_61, [16, 512, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_225: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_224, [16, 512, -1, 64]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_114: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_173, [1, 0])
        addmm_62: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_174, view_220, permute_114);  primals_174 = permute_114 = None
        view_227: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_62, [16, 512, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_228: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_227, [16, 512, -1, 64]);  view_227 = None

        # No stacktrace found for following nodes
        permute_default_78: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        permute_default_79: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        permute_default_80: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_78, permute_default_79, permute_default_80, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_191: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_13[0]

        # No stacktrace found for following nodes
        getitem_192: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_13[1]
        getitem_193: "i64[]" = _scaled_dot_product_efficient_attention_default_13[2]
        getitem_194: "i64[]" = _scaled_dot_product_efficient_attention_default_13[3];  _scaled_dot_product_efficient_attention_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_117: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_191, [0, 2, 1, 3])
        clone_43: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_235: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_43, [16, 512, 1024]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_235, [8192, 1024]);  view_235 = None
        permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_175, [1, 0])
        addmm_63: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_176, view_236, permute_118);  primals_176 = permute_118 = None
        view_237: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_63, [16, 512, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_27: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_32: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_137: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_138: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_137, 1.1111111111111112);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_85: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_81, mul_138);  add_81 = mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42: "f32[16, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[16, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_86: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  getitem_43 = None
        mul_139: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
        mul_140: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_139, primals_177)
        add_87: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_140, primals_178);  mul_140 = primals_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_87, [8192, 1024]);  add_87 = None
        permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_179, [1, 0])
        addmm_64: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_180, view_238, permute_119);  primals_180 = permute_119 = None
        view_239: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_64, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_141: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_142: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_143: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_141, add_88);  mul_141 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_143, [8192, 4096]);  mul_143 = None
        permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_181, [1, 0])
        addmm_65: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_182, view_240, permute_120);  primals_182 = permute_120 = None
        view_241: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_65, [16, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_26: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_33: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_144: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_145: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_89: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_85, mul_145);  add_85 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[16, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[16, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  getitem_45 = None
        mul_146: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
        mul_147: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_146, primals_183)
        add_91: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_147, primals_184);  mul_147 = primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_242: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_91, [8192, 1024]);  add_91 = None
        permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_185, [1, 0])
        addmm_66: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_186, view_242, permute_121);  primals_186 = permute_121 = None
        view_243: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_66, [16, 512, 1024]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_244: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_243, [16, 512, -1, 64]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_123: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_187, [1, 0])
        addmm_67: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_188, view_242, permute_123);  primals_188 = permute_123 = None
        view_246: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_67, [16, 512, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_247: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_246, [16, 512, -1, 64]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_125: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_189, [1, 0])
        addmm_68: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_190, view_242, permute_125);  primals_190 = permute_125 = None
        view_249: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_68, [16, 512, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_250: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_249, [16, 512, -1, 64]);  view_249 = None

        # No stacktrace found for following nodes
        permute_default_72: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        permute_default_73: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        permute_default_74: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_72, permute_default_73, permute_default_74, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_184: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_12[0]

        # No stacktrace found for following nodes
        getitem_185: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_12[1]
        getitem_186: "i64[]" = _scaled_dot_product_efficient_attention_default_12[2]
        getitem_187: "i64[]" = _scaled_dot_product_efficient_attention_default_12[3];  _scaled_dot_product_efficient_attention_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_128: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3])
        clone_47: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_257: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_47, [16, 512, 1024]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_257, [8192, 1024]);  view_257 = None
        permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_191, [1, 0])
        addmm_69: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_192, view_258, permute_129);  primals_192 = permute_129 = None
        view_259: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_69, [16, 512, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_25: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_35: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_150: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_151: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_93: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_89, mul_151);  add_89 = mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46: "f32[16, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[16, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_94: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  getitem_47 = None
        mul_152: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
        mul_153: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_152, primals_193)
        add_95: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_153, primals_194);  mul_153 = primals_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_95, [8192, 1024]);  add_95 = None
        permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_195, [1, 0])
        addmm_70: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_196, view_260, permute_130);  primals_196 = permute_130 = None
        view_261: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_70, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_154: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_155: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_156: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_154, add_96);  mul_154 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_156, [8192, 4096]);  mul_156 = None
        permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_197, [1, 0])
        addmm_71: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_198, view_262, permute_131);  primals_198 = permute_131 = None
        view_263: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_71, [16, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_24: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_36: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_157: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_158: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_97: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_93, mul_158);  add_93 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[16, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[16, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  getitem_49 = None
        mul_159: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
        mul_160: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_159, primals_199)
        add_99: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_160, primals_200);  mul_160 = primals_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_264: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_99, [8192, 1024]);  add_99 = None
        permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_201, [1, 0])
        addmm_72: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_202, view_264, permute_132);  primals_202 = permute_132 = None
        view_265: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_72, [16, 512, 1024]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_266: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_265, [16, 512, -1, 64]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_134: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_203, [1, 0])
        addmm_73: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_204, view_264, permute_134);  primals_204 = permute_134 = None
        view_268: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_73, [16, 512, 1024]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_269: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_268, [16, 512, -1, 64]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_136: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_205, [1, 0])
        addmm_74: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_206, view_264, permute_136);  primals_206 = permute_136 = None
        view_271: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_74, [16, 512, 1024]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_272: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_271, [16, 512, -1, 64]);  view_271 = None

        # No stacktrace found for following nodes
        permute_default_66: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        permute_default_67: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        permute_default_68: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_66, permute_default_67, permute_default_68, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_177: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0]

        # No stacktrace found for following nodes
        getitem_178: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_11[1]
        getitem_179: "i64[]" = _scaled_dot_product_efficient_attention_default_11[2]
        getitem_180: "i64[]" = _scaled_dot_product_efficient_attention_default_11[3];  _scaled_dot_product_efficient_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_139: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3])
        clone_51: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_279: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_51, [16, 512, 1024]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_279, [8192, 1024]);  view_279 = None
        permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_207, [1, 0])
        addmm_75: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_208, view_280, permute_140);  primals_208 = permute_140 = None
        view_281: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_75, [16, 512, 1024]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_23: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_38: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_163: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_38, view_281);  view_281 = None
        mul_164: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_101: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_97, mul_164);  add_97 = mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_50: "f32[16, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[16, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_102: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_39: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_101, getitem_51);  getitem_51 = None
        mul_165: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
        mul_166: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_165, primals_209)
        add_103: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_166, primals_210);  mul_166 = primals_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_103, [8192, 1024]);  add_103 = None
        permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_211, [1, 0])
        addmm_76: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_212, view_282, permute_141);  primals_212 = permute_141 = None
        view_283: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_76, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_167: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_168: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476);  view_283 = None
        erf_12: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_104: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_169: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_167, add_104);  mul_167 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_169, [8192, 4096]);  mul_169 = None
        permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_213, [1, 0])
        addmm_77: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_214, view_284, permute_142);  primals_214 = permute_142 = None
        view_285: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_77, [16, 512, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_22: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_39: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_170: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_39, view_285);  view_285 = None
        mul_171: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_170, 1.1111111111111112);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_105: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_101, mul_171);  add_101 = mul_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_52: "f32[16, 512, 1]" = var_mean_26[0]
        getitem_53: "f32[16, 512, 1]" = var_mean_26[1];  var_mean_26 = None
        add_106: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_40: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_53);  getitem_53 = None
        mul_172: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = None
        mul_173: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_172, primals_215)
        add_107: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_173, primals_216);  mul_173 = primals_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_286: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_107, [8192, 1024]);  add_107 = None
        permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_217, [1, 0])
        addmm_78: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_218, view_286, permute_143);  primals_218 = permute_143 = None
        view_287: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_78, [16, 512, 1024]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_288: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_287, [16, 512, -1, 64]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_145: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_219, [1, 0])
        addmm_79: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_220, view_286, permute_145);  primals_220 = permute_145 = None
        view_290: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_79, [16, 512, 1024]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_291: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_290, [16, 512, -1, 64]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_147: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_221, [1, 0])
        addmm_80: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_222, view_286, permute_147);  primals_222 = permute_147 = None
        view_293: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_80, [16, 512, 1024]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_294: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_293, [16, 512, -1, 64]);  view_293 = None

        # No stacktrace found for following nodes
        permute_default_60: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        permute_default_61: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        permute_default_62: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_60, permute_default_61, permute_default_62, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_170: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0]

        # No stacktrace found for following nodes
        getitem_171: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_10[1]
        getitem_172: "i64[]" = _scaled_dot_product_efficient_attention_default_10[2]
        getitem_173: "i64[]" = _scaled_dot_product_efficient_attention_default_10[3];  _scaled_dot_product_efficient_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_150: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3])
        clone_55: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_301: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_55, [16, 512, 1024]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_302: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_301, [8192, 1024]);  view_301 = None
        permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_223, [1, 0])
        addmm_81: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_224, view_302, permute_151);  primals_224 = permute_151 = None
        view_303: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_81, [16, 512, 1024]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_21: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_41: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_176: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_41, view_303);  view_303 = None
        mul_177: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_176, 1.1111111111111112);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_109: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_105, mul_177);  add_105 = mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_54: "f32[16, 512, 1]" = var_mean_27[0]
        getitem_55: "f32[16, 512, 1]" = var_mean_27[1];  var_mean_27 = None
        add_110: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_42: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_55);  getitem_55 = None
        mul_178: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = None
        mul_179: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_178, primals_225)
        add_111: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_179, primals_226);  mul_179 = primals_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_111, [8192, 1024]);  add_111 = None
        permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_227, [1, 0])
        addmm_82: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_228, view_304, permute_152);  primals_228 = permute_152 = None
        view_305: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_82, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_180: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_181: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476);  view_305 = None
        erf_13: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_112: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_182: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_180, add_112);  mul_180 = add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_306: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_182, [8192, 4096]);  mul_182 = None
        permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_229, [1, 0])
        addmm_83: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_230, view_306, permute_153);  primals_230 = permute_153 = None
        view_307: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_83, [16, 512, 1024]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_20: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_42: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_183: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_42, view_307);  view_307 = None
        mul_184: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_113: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_109, mul_184);  add_109 = mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_56: "f32[16, 512, 1]" = var_mean_28[0]
        getitem_57: "f32[16, 512, 1]" = var_mean_28[1];  var_mean_28 = None
        add_114: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_43: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_113, getitem_57);  getitem_57 = None
        mul_185: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = None
        mul_186: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_185, primals_231)
        add_115: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_186, primals_232);  mul_186 = primals_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_308: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_115, [8192, 1024]);  add_115 = None
        permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_233, [1, 0])
        addmm_84: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_234, view_308, permute_154);  primals_234 = permute_154 = None
        view_309: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_84, [16, 512, 1024]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_310: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_309, [16, 512, -1, 64]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_156: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_235, [1, 0])
        addmm_85: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_236, view_308, permute_156);  primals_236 = permute_156 = None
        view_312: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_85, [16, 512, 1024]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_313: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_312, [16, 512, -1, 64]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_158: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_237, [1, 0])
        addmm_86: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_238, view_308, permute_158);  primals_238 = permute_158 = None
        view_315: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_86, [16, 512, 1024]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_316: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_315, [16, 512, -1, 64]);  view_315 = None

        # No stacktrace found for following nodes
        permute_default_54: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        permute_default_55: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        permute_default_56: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_54, permute_default_55, permute_default_56, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_163: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0]

        # No stacktrace found for following nodes
        getitem_164: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_9[1]
        getitem_165: "i64[]" = _scaled_dot_product_efficient_attention_default_9[2]
        getitem_166: "i64[]" = _scaled_dot_product_efficient_attention_default_9[3];  _scaled_dot_product_efficient_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_161: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3])
        clone_59: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_323: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_59, [16, 512, 1024]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_323, [8192, 1024]);  view_323 = None
        permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_239, [1, 0])
        addmm_87: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_240, view_324, permute_162);  primals_240 = permute_162 = None
        view_325: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_87, [16, 512, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_19: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_44: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_189: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_44, view_325);  view_325 = None
        mul_190: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_117: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_113, mul_190);  add_113 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_58: "f32[16, 512, 1]" = var_mean_29[0]
        getitem_59: "f32[16, 512, 1]" = var_mean_29[1];  var_mean_29 = None
        add_118: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_45: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_59);  getitem_59 = None
        mul_191: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = None
        mul_192: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_191, primals_241)
        add_119: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_192, primals_242);  mul_192 = primals_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_326: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_119, [8192, 1024]);  add_119 = None
        permute_163: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_243, [1, 0])
        addmm_88: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_244, view_326, permute_163);  primals_244 = permute_163 = None
        view_327: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_88, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_193: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_194: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_14: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
        add_120: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_195: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_193, add_120);  mul_193 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_328: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_195, [8192, 4096]);  mul_195 = None
        permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_245, [1, 0])
        addmm_89: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_246, view_328, permute_164);  primals_246 = permute_164 = None
        view_329: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_89, [16, 512, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_18: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_45: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_196: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_45, view_329);  view_329 = None
        mul_197: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_196, 1.1111111111111112);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_121: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_117, mul_197);  add_117 = mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_60: "f32[16, 512, 1]" = var_mean_30[0]
        getitem_61: "f32[16, 512, 1]" = var_mean_30[1];  var_mean_30 = None
        add_122: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_46: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_121, getitem_61);  getitem_61 = None
        mul_198: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
        mul_199: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_198, primals_247)
        add_123: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_199, primals_248);  mul_199 = primals_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_330: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_123, [8192, 1024]);  add_123 = None
        permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_249, [1, 0])
        addmm_90: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_250, view_330, permute_165);  primals_250 = permute_165 = None
        view_331: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_90, [16, 512, 1024]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_332: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_331, [16, 512, -1, 64]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_167: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_251, [1, 0])
        addmm_91: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_252, view_330, permute_167);  primals_252 = permute_167 = None
        view_334: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_91, [16, 512, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_335: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_334, [16, 512, -1, 64]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_253, [1, 0])
        addmm_92: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_254, view_330, permute_169);  primals_254 = permute_169 = None
        view_337: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_92, [16, 512, 1024]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_338: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_337, [16, 512, -1, 64]);  view_337 = None

        # No stacktrace found for following nodes
        permute_default_48: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        permute_default_49: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        permute_default_50: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_48, permute_default_49, permute_default_50, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_156: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0]

        # No stacktrace found for following nodes
        getitem_157: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_8[1]
        getitem_158: "i64[]" = _scaled_dot_product_efficient_attention_default_8[2]
        getitem_159: "i64[]" = _scaled_dot_product_efficient_attention_default_8[3];  _scaled_dot_product_efficient_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_172: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3])
        clone_63: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_345: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_63, [16, 512, 1024]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_346: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_345, [8192, 1024]);  view_345 = None
        permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_255, [1, 0])
        addmm_93: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_256, view_346, permute_173);  primals_256 = permute_173 = None
        view_347: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_93, [16, 512, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_17: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_47: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_202: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_47, view_347);  view_347 = None
        mul_203: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_202, 1.1111111111111112);  mul_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_125: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_121, mul_203);  add_121 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_62: "f32[16, 512, 1]" = var_mean_31[0]
        getitem_63: "f32[16, 512, 1]" = var_mean_31[1];  var_mean_31 = None
        add_126: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_48: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_63);  getitem_63 = None
        mul_204: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = None
        mul_205: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_204, primals_257)
        add_127: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_205, primals_258);  mul_205 = primals_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_127, [8192, 1024]);  add_127 = None
        permute_174: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_259, [1, 0])
        addmm_94: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_260, view_348, permute_174);  primals_260 = permute_174 = None
        view_349: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_94, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_206: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_207: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
        erf_15: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_128: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_208: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_206, add_128);  mul_206 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_350: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_208, [8192, 4096]);  mul_208 = None
        permute_175: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_261, [1, 0])
        addmm_95: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_262, view_350, permute_175);  primals_262 = permute_175 = None
        view_351: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_95, [16, 512, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_16: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_48: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_209: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_48, view_351);  view_351 = None
        mul_210: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_209, 1.1111111111111112);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_129: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_125, mul_210);  add_125 = mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_64: "f32[16, 512, 1]" = var_mean_32[0]
        getitem_65: "f32[16, 512, 1]" = var_mean_32[1];  var_mean_32 = None
        add_130: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_49: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_129, getitem_65);  getitem_65 = None
        mul_211: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
        mul_212: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_211, primals_263)
        add_131: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_212, primals_264);  mul_212 = primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_352: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_131, [8192, 1024]);  add_131 = None
        permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_265, [1, 0])
        addmm_96: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_266, view_352, permute_176);  primals_266 = permute_176 = None
        view_353: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_96, [16, 512, 1024]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_354: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_353, [16, 512, -1, 64]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_178: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_267, [1, 0])
        addmm_97: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_268, view_352, permute_178);  primals_268 = permute_178 = None
        view_356: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_97, [16, 512, 1024]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_357: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_356, [16, 512, -1, 64]);  view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_269, [1, 0])
        addmm_98: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_270, view_352, permute_180);  primals_270 = permute_180 = None
        view_359: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_98, [16, 512, 1024]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_360: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_359, [16, 512, -1, 64]);  view_359 = None

        # No stacktrace found for following nodes
        permute_default_42: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        permute_default_43: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        permute_default_44: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_42, permute_default_43, permute_default_44, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_149: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0]

        # No stacktrace found for following nodes
        getitem_150: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_7[1]
        getitem_151: "i64[]" = _scaled_dot_product_efficient_attention_default_7[2]
        getitem_152: "i64[]" = _scaled_dot_product_efficient_attention_default_7[3];  _scaled_dot_product_efficient_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_183: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        clone_67: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_367: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_67, [16, 512, 1024]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_368: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_367, [8192, 1024]);  view_367 = None
        permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_271, [1, 0])
        addmm_99: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_272, view_368, permute_184);  primals_272 = permute_184 = None
        view_369: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_99, [16, 512, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_15: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_50: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_215: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_50, view_369);  view_369 = None
        mul_216: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_215, 1.1111111111111112);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_133: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_129, mul_216);  add_129 = mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_66: "f32[16, 512, 1]" = var_mean_33[0]
        getitem_67: "f32[16, 512, 1]" = var_mean_33[1];  var_mean_33 = None
        add_134: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_51: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_133, getitem_67);  getitem_67 = None
        mul_217: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = None
        mul_218: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_217, primals_273)
        add_135: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_218, primals_274);  mul_218 = primals_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_370: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_135, [8192, 1024]);  add_135 = None
        permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_275, [1, 0])
        addmm_100: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_276, view_370, permute_185);  primals_276 = permute_185 = None
        view_371: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_100, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_219: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_220: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
        erf_16: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
        add_136: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_221: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_219, add_136);  mul_219 = add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_372: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_221, [8192, 4096]);  mul_221 = None
        permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_277, [1, 0])
        addmm_101: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_278, view_372, permute_186);  primals_278 = permute_186 = None
        view_373: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_101, [16, 512, 1024]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_14: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_51: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_222: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_51, view_373);  view_373 = None
        mul_223: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_222, 1.1111111111111112);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_137: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_133, mul_223);  add_133 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_68: "f32[16, 512, 1]" = var_mean_34[0]
        getitem_69: "f32[16, 512, 1]" = var_mean_34[1];  var_mean_34 = None
        add_138: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_52: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_69);  getitem_69 = None
        mul_224: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
        mul_225: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_224, primals_279)
        add_139: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_225, primals_280);  mul_225 = primals_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_374: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_139, [8192, 1024]);  add_139 = None
        permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_281, [1, 0])
        addmm_102: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_282, view_374, permute_187);  primals_282 = permute_187 = None
        view_375: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_102, [16, 512, 1024]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_376: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_375, [16, 512, -1, 64]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_283, [1, 0])
        addmm_103: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_284, view_374, permute_189);  primals_284 = permute_189 = None
        view_378: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_103, [16, 512, 1024]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_379: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_378, [16, 512, -1, 64]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_191: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_285, [1, 0])
        addmm_104: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_286, view_374, permute_191);  primals_286 = permute_191 = None
        view_381: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_104, [16, 512, 1024]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_382: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_381, [16, 512, -1, 64]);  view_381 = None

        # No stacktrace found for following nodes
        permute_default_36: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        permute_default_37: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        permute_default_38: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_36, permute_default_37, permute_default_38, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_142: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0]

        # No stacktrace found for following nodes
        getitem_143: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_6[1]
        getitem_144: "i64[]" = _scaled_dot_product_efficient_attention_default_6[2]
        getitem_145: "i64[]" = _scaled_dot_product_efficient_attention_default_6[3];  _scaled_dot_product_efficient_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_194: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3])
        clone_71: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_389: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_71, [16, 512, 1024]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_389, [8192, 1024]);  view_389 = None
        permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_287, [1, 0])
        addmm_105: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_288, view_390, permute_195);  primals_288 = permute_195 = None
        view_391: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_105, [16, 512, 1024]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_13: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_53: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_228: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_53, view_391);  view_391 = None
        mul_229: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_228, 1.1111111111111112);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_141: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_137, mul_229);  add_137 = mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 512, 1]" = var_mean_35[0]
        getitem_71: "f32[16, 512, 1]" = var_mean_35[1];  var_mean_35 = None
        add_142: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_54: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_141, getitem_71);  getitem_71 = None
        mul_230: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = None
        mul_231: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_230, primals_289)
        add_143: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_231, primals_290);  mul_231 = primals_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_143, [8192, 1024]);  add_143 = None
        permute_196: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_291, [1, 0])
        addmm_106: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_292, view_392, permute_196);  primals_292 = permute_196 = None
        view_393: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_106, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_232: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_233: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
        erf_17: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_144: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_234: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_232, add_144);  mul_232 = add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_234, [8192, 4096]);  mul_234 = None
        permute_197: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_293, [1, 0])
        addmm_107: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_294, view_394, permute_197);  primals_294 = permute_197 = None
        view_395: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_107, [16, 512, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_12: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_54: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_235: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_54, view_395);  view_395 = None
        mul_236: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_145: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_141, mul_236);  add_141 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_72: "f32[16, 512, 1]" = var_mean_36[0]
        getitem_73: "f32[16, 512, 1]" = var_mean_36[1];  var_mean_36 = None
        add_146: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_55: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_145, getitem_73);  getitem_73 = None
        mul_237: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = None
        mul_238: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_237, primals_295)
        add_147: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_238, primals_296);  mul_238 = primals_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_396: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_147, [8192, 1024]);  add_147 = None
        permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_297, [1, 0])
        addmm_108: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_298, view_396, permute_198);  primals_298 = permute_198 = None
        view_397: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_108, [16, 512, 1024]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_398: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_397, [16, 512, -1, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_299, [1, 0])
        addmm_109: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_300, view_396, permute_200);  primals_300 = permute_200 = None
        view_400: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_109, [16, 512, 1024]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_401: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_400, [16, 512, -1, 64]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_301, [1, 0])
        addmm_110: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_302, view_396, permute_202);  primals_302 = permute_202 = None
        view_403: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_110, [16, 512, 1024]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_404: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_403, [16, 512, -1, 64]);  view_403 = None

        # No stacktrace found for following nodes
        permute_default_30: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        permute_default_31: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
        permute_default_32: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_135: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0]

        # No stacktrace found for following nodes
        getitem_136: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_5[1]
        getitem_137: "i64[]" = _scaled_dot_product_efficient_attention_default_5[2]
        getitem_138: "i64[]" = _scaled_dot_product_efficient_attention_default_5[3];  _scaled_dot_product_efficient_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_205: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3])
        clone_75: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_411: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_75, [16, 512, 1024]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_412: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_411, [8192, 1024]);  view_411 = None
        permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_303, [1, 0])
        addmm_111: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_304, view_412, permute_206);  primals_304 = permute_206 = None
        view_413: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_111, [16, 512, 1024]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_11: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        gt_56: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_241: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_56, view_413);  view_413 = None
        mul_242: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_241, 1.1111111111111112);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_149: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_145, mul_242);  add_145 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_74: "f32[16, 512, 1]" = var_mean_37[0]
        getitem_75: "f32[16, 512, 1]" = var_mean_37[1];  var_mean_37 = None
        add_150: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_57: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_75);  getitem_75 = None
        mul_243: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = None
        mul_244: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_243, primals_305)
        add_151: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_244, primals_306);  mul_244 = primals_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_414: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_151, [8192, 1024]);  add_151 = None
        permute_207: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_307, [1, 0])
        addmm_112: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_308, view_414, permute_207);  primals_308 = permute_207 = None
        view_415: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_112, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_245: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_246: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
        erf_18: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_152: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_247: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_245, add_152);  mul_245 = add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_416: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_247, [8192, 4096]);  mul_247 = None
        permute_208: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_309, [1, 0])
        addmm_113: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_310, view_416, permute_208);  primals_310 = permute_208 = None
        view_417: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_113, [16, 512, 1024]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_10: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_57: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_248: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_57, view_417);  view_417 = None
        mul_249: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_248, 1.1111111111111112);  mul_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_153: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_149, mul_249);  add_149 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_76: "f32[16, 512, 1]" = var_mean_38[0]
        getitem_77: "f32[16, 512, 1]" = var_mean_38[1];  var_mean_38 = None
        add_154: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_58: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_77);  getitem_77 = None
        mul_250: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = None
        mul_251: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_250, primals_311)
        add_155: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_251, primals_312);  mul_251 = primals_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_418: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_155, [8192, 1024]);  add_155 = None
        permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_313, [1, 0])
        addmm_114: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_314, view_418, permute_209);  primals_314 = permute_209 = None
        view_419: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_114, [16, 512, 1024]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_420: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_419, [16, 512, -1, 64]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_211: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_315, [1, 0])
        addmm_115: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_316, view_418, permute_211);  primals_316 = permute_211 = None
        view_422: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_115, [16, 512, 1024]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_423: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_422, [16, 512, -1, 64]);  view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_317, [1, 0])
        addmm_116: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_318, view_418, permute_213);  primals_318 = permute_213 = None
        view_425: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_116, [16, 512, 1024]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_426: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_425, [16, 512, -1, 64]);  view_425 = None

        # No stacktrace found for following nodes
        permute_default_24: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        permute_default_25: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
        permute_default_26: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_128: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0]

        # No stacktrace found for following nodes
        getitem_129: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_4[1]
        getitem_130: "i64[]" = _scaled_dot_product_efficient_attention_default_4[2]
        getitem_131: "i64[]" = _scaled_dot_product_efficient_attention_default_4[3];  _scaled_dot_product_efficient_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_216: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3])
        clone_79: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_433: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_79, [16, 512, 1024]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_434: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_433, [8192, 1024]);  view_433 = None
        permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_319, [1, 0])
        addmm_117: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_320, view_434, permute_217);  primals_320 = permute_217 = None
        view_435: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_117, [16, 512, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_9: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_59: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_254: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_59, view_435);  view_435 = None
        mul_255: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_254, 1.1111111111111112);  mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_157: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_153, mul_255);  add_153 = mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 512, 1]" = var_mean_39[0]
        getitem_79: "f32[16, 512, 1]" = var_mean_39[1];  var_mean_39 = None
        add_158: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_60: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_79);  getitem_79 = None
        mul_256: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = None
        mul_257: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_256, primals_321)
        add_159: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_257, primals_322);  mul_257 = primals_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_159, [8192, 1024]);  add_159 = None
        permute_218: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_323, [1, 0])
        addmm_118: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_324, view_436, permute_218);  primals_324 = permute_218 = None
        view_437: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_118, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_258: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_259: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476);  view_437 = None
        erf_19: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_259);  mul_259 = None
        add_160: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_260: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_258, add_160);  mul_258 = add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_438: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_260, [8192, 4096]);  mul_260 = None
        permute_219: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_325, [1, 0])
        addmm_119: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_326, view_438, permute_219);  primals_326 = permute_219 = None
        view_439: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_119, [16, 512, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_8: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_60: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_261: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_60, view_439);  view_439 = None
        mul_262: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_261, 1.1111111111111112);  mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_161: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_157, mul_262);  add_157 = mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_80: "f32[16, 512, 1]" = var_mean_40[0]
        getitem_81: "f32[16, 512, 1]" = var_mean_40[1];  var_mean_40 = None
        add_162: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_61: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_161, getitem_81);  getitem_81 = None
        mul_263: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = None
        mul_264: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_263, primals_327)
        add_163: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_264, primals_328);  mul_264 = primals_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_440: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_163, [8192, 1024]);  add_163 = None
        permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_329, [1, 0])
        addmm_120: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_330, view_440, permute_220);  primals_330 = permute_220 = None
        view_441: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_120, [16, 512, 1024]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_442: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_441, [16, 512, -1, 64]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_331, [1, 0])
        addmm_121: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_332, view_440, permute_222);  primals_332 = permute_222 = None
        view_444: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_121, [16, 512, 1024]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_445: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_444, [16, 512, -1, 64]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_333, [1, 0])
        addmm_122: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_334, view_440, permute_224);  primals_334 = permute_224 = None
        view_447: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_122, [16, 512, 1024]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_448: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_447, [16, 512, -1, 64]);  view_447 = None

        # No stacktrace found for following nodes
        permute_default_18: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        permute_default_19: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        permute_default_20: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_121: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0]

        # No stacktrace found for following nodes
        getitem_122: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_3[1]
        getitem_123: "i64[]" = _scaled_dot_product_efficient_attention_default_3[2]
        getitem_124: "i64[]" = _scaled_dot_product_efficient_attention_default_3[3];  _scaled_dot_product_efficient_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_227: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3])
        clone_83: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_455: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_83, [16, 512, 1024]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_455, [8192, 1024]);  view_455 = None
        permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_335, [1, 0])
        addmm_123: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_336, view_456, permute_228);  primals_336 = permute_228 = None
        view_457: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_123, [16, 512, 1024]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_7: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_62: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_267: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_62, view_457);  view_457 = None
        mul_268: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_267, 1.1111111111111112);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_165: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_161, mul_268);  add_161 = mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_82: "f32[16, 512, 1]" = var_mean_41[0]
        getitem_83: "f32[16, 512, 1]" = var_mean_41[1];  var_mean_41 = None
        add_166: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_63: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_165, getitem_83);  getitem_83 = None
        mul_269: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = None
        mul_270: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_269, primals_337)
        add_167: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_270, primals_338);  mul_270 = primals_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_167, [8192, 1024]);  add_167 = None
        permute_229: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_339, [1, 0])
        addmm_124: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_340, view_458, permute_229);  primals_340 = permute_229 = None
        view_459: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_124, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_271: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_272: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476);  view_459 = None
        erf_20: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_272);  mul_272 = None
        add_168: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_273: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_271, add_168);  mul_271 = add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_460: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_273, [8192, 4096]);  mul_273 = None
        permute_230: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_341, [1, 0])
        addmm_125: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_342, view_460, permute_230);  primals_342 = permute_230 = None
        view_461: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_125, [16, 512, 1024]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_6: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_63: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_274: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_63, view_461);  view_461 = None
        mul_275: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_274, 1.1111111111111112);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_169: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_165, mul_275);  add_165 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_84: "f32[16, 512, 1]" = var_mean_42[0]
        getitem_85: "f32[16, 512, 1]" = var_mean_42[1];  var_mean_42 = None
        add_170: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_64: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_85);  getitem_85 = None
        mul_276: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = None
        mul_277: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_276, primals_343)
        add_171: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_277, primals_344);  mul_277 = primals_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_462: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_171, [8192, 1024]);  add_171 = None
        permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_345, [1, 0])
        addmm_126: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_346, view_462, permute_231);  primals_346 = permute_231 = None
        view_463: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_126, [16, 512, 1024]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_464: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_463, [16, 512, -1, 64]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_347, [1, 0])
        addmm_127: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_348, view_462, permute_233);  primals_348 = permute_233 = None
        view_466: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_127, [16, 512, 1024]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_467: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_466, [16, 512, -1, 64]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_349, [1, 0])
        addmm_128: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_350, view_462, permute_235);  primals_350 = permute_235 = None
        view_469: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_128, [16, 512, 1024]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_470: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_469, [16, 512, -1, 64]);  view_469 = None

        # No stacktrace found for following nodes
        permute_default_12: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        permute_default_13: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
        permute_default_14: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_114: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0]

        # No stacktrace found for following nodes
        getitem_115: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_2[1]
        getitem_116: "i64[]" = _scaled_dot_product_efficient_attention_default_2[2]
        getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_default_2[3];  _scaled_dot_product_efficient_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_238: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3])
        clone_87: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_477: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_87, [16, 512, 1024]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_477, [8192, 1024]);  view_477 = None
        permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_351, [1, 0])
        addmm_129: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_352, view_478, permute_239);  primals_352 = permute_239 = None
        view_479: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_129, [16, 512, 1024]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_5: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        gt_65: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_280: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_65, view_479);  view_479 = None
        mul_281: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_280, 1.1111111111111112);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_173: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_169, mul_281);  add_169 = mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_86: "f32[16, 512, 1]" = var_mean_43[0]
        getitem_87: "f32[16, 512, 1]" = var_mean_43[1];  var_mean_43 = None
        add_174: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_66: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_173, getitem_87);  getitem_87 = None
        mul_282: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = None
        mul_283: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_282, primals_353)
        add_175: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_283, primals_354);  mul_283 = primals_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_480: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_175, [8192, 1024]);  add_175 = None
        permute_240: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_355, [1, 0])
        addmm_130: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_356, view_480, permute_240);  primals_356 = permute_240 = None
        view_481: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_130, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_284: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_285: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476);  view_481 = None
        erf_21: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
        add_176: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_286: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_284, add_176);  mul_284 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_482: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_286, [8192, 4096]);  mul_286 = None
        permute_241: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_357, [1, 0])
        addmm_131: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_358, view_482, permute_241);  primals_358 = permute_241 = None
        view_483: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_131, [16, 512, 1024]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_4: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        gt_66: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_287: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_66, view_483);  view_483 = None
        mul_288: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_287, 1.1111111111111112);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_177: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_173, mul_288);  add_173 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_88: "f32[16, 512, 1]" = var_mean_44[0]
        getitem_89: "f32[16, 512, 1]" = var_mean_44[1];  var_mean_44 = None
        add_178: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_67: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_89);  getitem_89 = None
        mul_289: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = None
        mul_290: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_289, primals_359)
        add_179: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_290, primals_360);  mul_290 = primals_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_484: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_179, [8192, 1024]);  add_179 = None
        permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_361, [1, 0])
        addmm_132: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_362, view_484, permute_242);  primals_362 = permute_242 = None
        view_485: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_132, [16, 512, 1024]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_486: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_485, [16, 512, -1, 64]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_363, [1, 0])
        addmm_133: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_364, view_484, permute_244);  primals_364 = permute_244 = None
        view_488: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_133, [16, 512, 1024]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_489: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_488, [16, 512, -1, 64]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_246: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_365, [1, 0])
        addmm_134: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_366, view_484, permute_246);  primals_366 = permute_246 = None
        view_491: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_134, [16, 512, 1024]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_492: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_491, [16, 512, -1, 64]);  view_491 = None

        # No stacktrace found for following nodes
        permute_default_6: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        permute_default_7: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        permute_default_8: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_107: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0]

        # No stacktrace found for following nodes
        getitem_108: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default_1[1]
        getitem_109: "i64[]" = _scaled_dot_product_efficient_attention_default_1[2]
        getitem_110: "i64[]" = _scaled_dot_product_efficient_attention_default_1[3];  _scaled_dot_product_efficient_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_249: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_107, [0, 2, 1, 3])
        clone_91: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_499: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_91, [16, 512, 1024]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_499, [8192, 1024]);  view_499 = None
        permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_367, [1, 0])
        addmm_135: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_368, view_500, permute_250);  primals_368 = permute_250 = None
        view_501: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_135, [16, 512, 1024]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_3: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_68: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_293: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_68, view_501);  view_501 = None
        mul_294: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_293, 1.1111111111111112);  mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_181: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_177, mul_294);  add_177 = mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
        getitem_90: "f32[16, 512, 1]" = var_mean_45[0]
        getitem_91: "f32[16, 512, 1]" = var_mean_45[1];  var_mean_45 = None
        add_182: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_69: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_181, getitem_91);  getitem_91 = None
        mul_295: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = None
        mul_296: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_295, primals_369)
        add_183: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_296, primals_370);  mul_296 = primals_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_183, [8192, 1024]);  add_183 = None
        permute_251: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_371, [1, 0])
        addmm_136: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_372, view_502, permute_251);  primals_372 = permute_251 = None
        view_503: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_136, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_297: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_298: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
        erf_22: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
        add_184: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_299: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_297, add_184);  mul_297 = add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_299, [8192, 4096]);  mul_299 = None
        permute_252: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_373, [1, 0])
        addmm_137: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_374, view_504, permute_252);  primals_374 = permute_252 = None
        view_505: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_137, [16, 512, 1024]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_2: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_69: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_300: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_69, view_505);  view_505 = None
        mul_301: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_300, 1.1111111111111112);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_185: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_181, mul_301);  add_181 = mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_92: "f32[16, 512, 1]" = var_mean_46[0]
        getitem_93: "f32[16, 512, 1]" = var_mean_46[1];  var_mean_46 = None
        add_186: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_70: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_185, getitem_93);  getitem_93 = None
        mul_302: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = None
        mul_303: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_302, primals_375)
        add_187: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_303, primals_376);  mul_303 = primals_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_506: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_187, [8192, 1024]);  add_187 = None
        permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_377, [1, 0])
        addmm_138: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_378, view_506, permute_253);  primals_378 = permute_253 = None
        view_507: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_138, [16, 512, 1024]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_508: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_507, [16, 512, -1, 64]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_379, [1, 0])
        addmm_139: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_380, view_506, permute_255);  primals_380 = permute_255 = None
        view_510: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_139, [16, 512, 1024]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_511: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_510, [16, 512, -1, 64]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_257: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_381, [1, 0])
        addmm_140: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_382, view_506, permute_257);  primals_382 = permute_257 = None
        view_513: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_140, [16, 512, 1024]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_514: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_513, [16, 512, -1, 64]);  view_513 = None

        # No stacktrace found for following nodes
        permute_default: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        permute_default_1: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        permute_default_2: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, True, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_100: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default[0]

        # No stacktrace found for following nodes
        getitem_101: "f32[16, 16, 512]" = _scaled_dot_product_efficient_attention_default[1]
        getitem_102: "i64[]" = _scaled_dot_product_efficient_attention_default[2]
        getitem_103: "i64[]" = _scaled_dot_product_efficient_attention_default[3];  _scaled_dot_product_efficient_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_260: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3])
        clone_95: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_521: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_95, [16, 512, 1024]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_522: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_521, [8192, 1024]);  view_521 = None
        permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_383, [1, 0])
        addmm_141: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_384, view_522, permute_261);  primals_384 = permute_261 = None
        view_523: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_141, [16, 512, 1024]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_1: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        gt_71: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_306: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_71, view_523);  view_523 = None
        mul_307: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_306, 1.1111111111111112);  mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_189: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_185, mul_307);  add_185 = mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
        getitem_94: "f32[16, 512, 1]" = var_mean_47[0]
        getitem_95: "f32[16, 512, 1]" = var_mean_47[1];  var_mean_47 = None
        add_190: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_72: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_189, getitem_95);  getitem_95 = None
        mul_308: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = None
        mul_309: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_308, primals_385)
        add_191: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_309, primals_386);  mul_309 = primals_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_524: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_191, [8192, 1024]);  add_191 = None
        permute_262: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_387, [1, 0])
        addmm_142: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_388, view_524, permute_262);  primals_388 = permute_262 = None
        view_525: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_142, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_310: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_311: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476);  view_525 = None
        erf_23: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_192: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_312: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_310, add_192);  mul_310 = add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_526: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_312, [8192, 4096]);  mul_312 = None
        permute_263: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_389, [1, 0])
        addmm_143: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_390, view_526, permute_263);  primals_390 = permute_263 = None
        view_527: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_143, [16, 512, 1024]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        gt_72: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_313: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_72, view_527);  view_527 = None
        mul_314: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_313, 1.1111111111111112);  mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_193: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_189, mul_314);  add_189 = mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 512, 1]" = var_mean_48[0]
        getitem_97: "f32[16, 512, 1]" = var_mean_48[1];  var_mean_48 = None
        add_194: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_48: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_73: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_97);  add_193 = getitem_97 = None
        mul_315: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = None
        mul_316: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_315, primals_391)
        add_195: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_316, primals_392);  mul_316 = primals_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        view_528: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_195, [8192, 1024]);  add_195 = None
        permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_393, [1, 0])
        addmm_144: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_394, view_528, permute_264);  primals_394 = permute_264 = None
        view_529: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_144, [16, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_317: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_318: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
        erf_24: "f32[16, 512, 1024]" = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_196: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_319: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_317, add_196);  mul_317 = add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_49 = torch.ops.aten.var_mean.correction(mul_319, [2], correction = 0, keepdim = True)
        getitem_98: "f32[16, 512, 1]" = var_mean_49[0]
        getitem_99: "f32[16, 512, 1]" = var_mean_49[1];  var_mean_49 = None
        add_197: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_49: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_74: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_319, getitem_99);  mul_319 = None
        mul_320: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
        mul_321: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_320, primals_395);  mul_320 = None
        add_198: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_321, primals_396);  mul_321 = primals_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        view_530: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_198, [8192, 1024]);  add_198 = None
        permute_265: "f32[1024, 29056]" = torch.ops.aten.permute.default(primals_3, [1, 0])
        addmm_145: "f32[8192, 29056]" = torch.ops.aten.addmm.default(primals_397, view_530, permute_265);  primals_397 = permute_265 = None
        view_531: "f32[16, 512, 29056]" = torch.ops.aten.reshape.default(addmm_145, [16, 512, 29056]);  addmm_145 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[16, 513]" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1], -100.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[16, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone_96: "i64[16, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_532: "f32[8192, 29056]" = torch.ops.aten.reshape.default(view_531, [-1, 29056])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_533: "i64[8192]" = torch.ops.aten.reshape.default(clone_96, [-1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_24: "f32[8192, 1]" = torch.ops.aten.amax.default(view_532, [1], True)
        sub_75: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = None
        exp_24: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne, view_533, full_default_2);  view_533 = full_default_2 = None
        unsqueeze_2: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_2);  sub_76 = unsqueeze_2 = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192]" = torch.ops.aten.where.self(ne, neg, full_default_3);  neg = full_default_3 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        div_51: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_52: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_54: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_55: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_57: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_58: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_60: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_61: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_63: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_64: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_66: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_67: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_69: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_70: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_72: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_73: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_75: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_76: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_78: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_79: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_81: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_82: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_84: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_85: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_87: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_88: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_90: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_91: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_93: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_94: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_96: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_97: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_99: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_100: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_102: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_103: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_105: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_106: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_108: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_109: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_111: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_112: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_114: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_115: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_117: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_118: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_120: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_121: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_123: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        return (div_48, view_531, primals_2, primals_3, primals_4, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_295, primals_297, primals_299, primals_301, primals_303, primals_305, primals_307, primals_309, primals_311, primals_313, primals_315, primals_317, primals_319, primals_321, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_335, primals_337, primals_339, primals_341, primals_343, primals_345, primals_347, primals_349, primals_351, primals_353, primals_355, primals_357, primals_359, primals_361, primals_363, primals_365, primals_367, primals_369, primals_371, primals_373, primals_375, primals_377, primals_379, primals_381, primals_383, primals_385, primals_387, primals_389, primals_391, primals_393, primals_395, full_default, gt, mul_3, view, permute_default_138, permute_default_139, permute_default_140, getitem_261, getitem_262, getitem_263, getitem_264, view_16, gt_2, mul_9, view_18, addmm_4, view_20, gt_3, mul_16, view_22, permute_default_132, permute_default_133, permute_default_134, getitem_254, getitem_255, getitem_256, getitem_257, view_38, gt_5, mul_22, view_40, addmm_10, view_42, gt_6, mul_29, view_44, permute_default_126, permute_default_127, permute_default_128, getitem_247, getitem_248, getitem_249, getitem_250, view_60, gt_8, mul_35, view_62, addmm_16, view_64, gt_9, mul_42, view_66, permute_default_120, permute_default_121, permute_default_122, getitem_240, getitem_241, getitem_242, getitem_243, view_82, gt_11, mul_48, view_84, addmm_22, view_86, gt_12, mul_55, view_88, permute_default_114, permute_default_115, permute_default_116, getitem_233, getitem_234, getitem_235, getitem_236, view_104, gt_14, mul_61, view_106, addmm_28, view_108, gt_15, mul_68, view_110, permute_default_108, permute_default_109, permute_default_110, getitem_226, getitem_227, getitem_228, getitem_229, view_126, gt_17, mul_74, view_128, addmm_34, view_130, gt_18, mul_81, view_132, permute_default_102, permute_default_103, permute_default_104, getitem_219, getitem_220, getitem_221, getitem_222, view_148, gt_20, mul_87, view_150, addmm_40, view_152, gt_21, mul_94, view_154, permute_default_96, permute_default_97, permute_default_98, getitem_212, getitem_213, getitem_214, getitem_215, view_170, gt_23, mul_100, view_172, addmm_46, view_174, gt_24, mul_107, view_176, permute_default_90, permute_default_91, permute_default_92, getitem_205, getitem_206, getitem_207, getitem_208, view_192, gt_26, mul_113, view_194, addmm_52, view_196, gt_27, mul_120, view_198, permute_default_84, permute_default_85, permute_default_86, getitem_198, getitem_199, getitem_200, getitem_201, view_214, gt_29, mul_126, view_216, addmm_58, view_218, gt_30, mul_133, view_220, permute_default_78, permute_default_79, permute_default_80, getitem_191, getitem_192, getitem_193, getitem_194, view_236, gt_32, mul_139, view_238, addmm_64, view_240, gt_33, mul_146, view_242, permute_default_72, permute_default_73, permute_default_74, getitem_184, getitem_185, getitem_186, getitem_187, view_258, gt_35, mul_152, view_260, addmm_70, view_262, gt_36, mul_159, view_264, permute_default_66, permute_default_67, permute_default_68, getitem_177, getitem_178, getitem_179, getitem_180, view_280, gt_38, mul_165, view_282, addmm_76, view_284, gt_39, mul_172, view_286, permute_default_60, permute_default_61, permute_default_62, getitem_170, getitem_171, getitem_172, getitem_173, view_302, gt_41, mul_178, view_304, addmm_82, view_306, gt_42, mul_185, view_308, permute_default_54, permute_default_55, permute_default_56, getitem_163, getitem_164, getitem_165, getitem_166, view_324, gt_44, mul_191, view_326, addmm_88, view_328, gt_45, mul_198, view_330, permute_default_48, permute_default_49, permute_default_50, getitem_156, getitem_157, getitem_158, getitem_159, view_346, gt_47, mul_204, view_348, addmm_94, view_350, gt_48, mul_211, view_352, permute_default_42, permute_default_43, permute_default_44, getitem_149, getitem_150, getitem_151, getitem_152, view_368, gt_50, mul_217, view_370, addmm_100, view_372, gt_51, mul_224, view_374, permute_default_36, permute_default_37, permute_default_38, getitem_142, getitem_143, getitem_144, getitem_145, view_390, gt_53, mul_230, view_392, addmm_106, view_394, gt_54, mul_237, view_396, permute_default_30, permute_default_31, permute_default_32, getitem_135, getitem_136, getitem_137, getitem_138, view_412, gt_56, mul_243, view_414, addmm_112, view_416, gt_57, mul_250, view_418, permute_default_24, permute_default_25, permute_default_26, getitem_128, getitem_129, getitem_130, getitem_131, view_434, gt_59, mul_256, view_436, addmm_118, view_438, gt_60, mul_263, view_440, permute_default_18, permute_default_19, permute_default_20, getitem_121, getitem_122, getitem_123, getitem_124, view_456, gt_62, mul_269, view_458, addmm_124, view_460, gt_63, mul_276, view_462, permute_default_12, permute_default_13, permute_default_14, getitem_114, getitem_115, getitem_116, getitem_117, view_478, gt_65, mul_282, view_480, addmm_130, view_482, gt_66, mul_289, view_484, permute_default_6, permute_default_7, permute_default_8, getitem_107, getitem_108, getitem_109, getitem_110, view_500, gt_68, mul_295, view_502, addmm_136, view_504, gt_69, mul_302, view_506, permute_default, permute_default_1, permute_default_2, getitem_100, getitem_101, getitem_102, getitem_103, view_522, gt_71, mul_308, view_524, addmm_142, view_526, gt_72, mul_315, view_528, addmm_144, getitem_99, rsqrt_49, view_530, view_531, constant_pad_nd, amax_24, log, convert_element_type, div_51, div_52, div_54, div_55, div_57, div_58, div_60, div_61, div_63, div_64, div_66, div_67, div_69, div_70, div_72, div_73, div_75, div_76, div_78, div_79, div_81, div_82, div_84, div_85, div_87, div_88, div_90, div_91, div_93, div_94, div_96, div_97, div_99, div_100, div_102, div_103, div_105, div_106, div_108, div_109, div_111, div_112, div_114, div_115, div_117, div_118, div_120, div_121, div_123)
