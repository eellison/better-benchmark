import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "i64[1, 512]", primals_4: "f32[30522, 768]", primals_7: "f32[768]", primals_9: "f32[768, 768]", primals_11: "f32[768, 768]", primals_13: "f32[768, 768]", primals_15: "f32[768, 768]", primals_17: "f32[768]", primals_19: "f32[3072, 768]", primals_21: "f32[768, 3072]", primals_23: "f32[768]", primals_25: "f32[768, 768]", primals_27: "f32[768, 768]", primals_29: "f32[768, 768]", primals_31: "f32[768, 768]", primals_33: "f32[768]", primals_35: "f32[3072, 768]", primals_37: "f32[768, 3072]", primals_39: "f32[768]", primals_41: "f32[768, 768]", primals_43: "f32[768, 768]", primals_45: "f32[768, 768]", primals_47: "f32[768, 768]", primals_49: "f32[768]", primals_51: "f32[3072, 768]", primals_53: "f32[768, 3072]", primals_55: "f32[768]", primals_57: "f32[768, 768]", primals_59: "f32[768, 768]", primals_61: "f32[768, 768]", primals_63: "f32[768, 768]", primals_65: "f32[768]", primals_67: "f32[3072, 768]", primals_69: "f32[768, 3072]", primals_71: "f32[768]", primals_73: "f32[768, 768]", primals_75: "f32[768, 768]", primals_77: "f32[768, 768]", primals_79: "f32[768, 768]", primals_81: "f32[768]", primals_83: "f32[3072, 768]", primals_85: "f32[768, 3072]", primals_87: "f32[768]", primals_89: "f32[768, 768]", primals_91: "f32[768, 768]", primals_93: "f32[768, 768]", primals_95: "f32[768, 768]", primals_97: "f32[768]", primals_99: "f32[3072, 768]", primals_101: "f32[768, 3072]", primals_103: "f32[768]", primals_105: "f32[768, 768]", primals_107: "f32[768, 768]", primals_109: "f32[768, 768]", primals_111: "f32[768, 768]", primals_113: "f32[768]", primals_115: "f32[3072, 768]", primals_117: "f32[768, 3072]", primals_119: "f32[768]", primals_121: "f32[768, 768]", primals_123: "f32[768, 768]", primals_125: "f32[768, 768]", primals_127: "f32[768, 768]", primals_129: "f32[768]", primals_131: "f32[3072, 768]", primals_133: "f32[768, 3072]", primals_135: "f32[768]", primals_137: "f32[768, 768]", primals_139: "f32[768, 768]", primals_141: "f32[768, 768]", primals_143: "f32[768, 768]", primals_145: "f32[768]", primals_147: "f32[3072, 768]", primals_149: "f32[768, 3072]", primals_151: "f32[768]", primals_153: "f32[768, 768]", primals_155: "f32[768, 768]", primals_157: "f32[768, 768]", primals_159: "f32[768, 768]", primals_161: "f32[768]", primals_163: "f32[3072, 768]", primals_165: "f32[768, 3072]", primals_167: "f32[768]", primals_169: "f32[768, 768]", primals_171: "f32[768, 768]", primals_173: "f32[768, 768]", primals_175: "f32[768, 768]", primals_177: "f32[768]", primals_179: "f32[3072, 768]", primals_181: "f32[768, 3072]", primals_183: "f32[768]", primals_185: "f32[768, 768]", primals_187: "f32[768, 768]", primals_189: "f32[768, 768]", primals_191: "f32[768, 768]", primals_193: "f32[768]", primals_195: "f32[3072, 768]", primals_197: "f32[768, 3072]", primals_199: "f32[768]", primals_201: "f32[768, 768]", primals_203: "f32[768]", primals_206: "i64[32, 512]", gather: "i64[1, 512]", mul: "f32[32, 512, 768]", gt: "b8[32, 512, 768]", ge: "b8[1, 1, 512, 1]", view: "f32[16384, 768]", bmm: "f32[384, 512, 512]", amax: "f32[32, 12, 512, 1]", sum_1: "f32[32, 12, 512, 1]", logical_not_1: "b8[32, 12, 512, 1]", gt_1: "b8[32, 12, 512, 512]", view_16: "f32[16384, 768]", gt_2: "b8[32, 512, 768]", mul_10: "f32[32, 512, 768]", view_18: "f32[16384, 768]", addmm_4: "f32[16384, 3072]", view_20: "f32[16384, 3072]", gt_3: "b8[32, 512, 768]", mul_17: "f32[32, 512, 768]", view_22: "f32[16384, 768]", where_3: "f32[32, 12, 512, 512]", gt_4: "b8[32, 12, 512, 512]", view_38: "f32[16384, 768]", gt_5: "b8[32, 512, 768]", mul_25: "f32[32, 512, 768]", view_40: "f32[16384, 768]", addmm_10: "f32[16384, 3072]", view_42: "f32[16384, 3072]", gt_6: "b8[32, 512, 768]", mul_32: "f32[32, 512, 768]", view_44: "f32[16384, 768]", where_5: "f32[32, 12, 512, 512]", gt_7: "b8[32, 12, 512, 512]", view_60: "f32[16384, 768]", gt_8: "b8[32, 512, 768]", mul_40: "f32[32, 512, 768]", view_62: "f32[16384, 768]", addmm_16: "f32[16384, 3072]", view_64: "f32[16384, 3072]", gt_9: "b8[32, 512, 768]", mul_47: "f32[32, 512, 768]", view_66: "f32[16384, 768]", where_7: "f32[32, 12, 512, 512]", gt_10: "b8[32, 12, 512, 512]", view_82: "f32[16384, 768]", gt_11: "b8[32, 512, 768]", mul_55: "f32[32, 512, 768]", view_84: "f32[16384, 768]", addmm_22: "f32[16384, 3072]", view_86: "f32[16384, 3072]", gt_12: "b8[32, 512, 768]", mul_62: "f32[32, 512, 768]", view_88: "f32[16384, 768]", where_9: "f32[32, 12, 512, 512]", gt_13: "b8[32, 12, 512, 512]", view_104: "f32[16384, 768]", gt_14: "b8[32, 512, 768]", mul_70: "f32[32, 512, 768]", view_106: "f32[16384, 768]", addmm_28: "f32[16384, 3072]", view_108: "f32[16384, 3072]", gt_15: "b8[32, 512, 768]", mul_77: "f32[32, 512, 768]", view_110: "f32[16384, 768]", where_11: "f32[32, 12, 512, 512]", gt_16: "b8[32, 12, 512, 512]", view_126: "f32[16384, 768]", gt_17: "b8[32, 512, 768]", mul_85: "f32[32, 512, 768]", view_128: "f32[16384, 768]", addmm_34: "f32[16384, 3072]", view_130: "f32[16384, 3072]", gt_18: "b8[32, 512, 768]", mul_92: "f32[32, 512, 768]", view_132: "f32[16384, 768]", where_13: "f32[32, 12, 512, 512]", gt_19: "b8[32, 12, 512, 512]", view_148: "f32[16384, 768]", gt_20: "b8[32, 512, 768]", mul_100: "f32[32, 512, 768]", view_150: "f32[16384, 768]", addmm_40: "f32[16384, 3072]", view_152: "f32[16384, 3072]", gt_21: "b8[32, 512, 768]", mul_107: "f32[32, 512, 768]", view_154: "f32[16384, 768]", where_15: "f32[32, 12, 512, 512]", gt_22: "b8[32, 12, 512, 512]", view_170: "f32[16384, 768]", gt_23: "b8[32, 512, 768]", mul_115: "f32[32, 512, 768]", view_172: "f32[16384, 768]", addmm_46: "f32[16384, 3072]", view_174: "f32[16384, 3072]", gt_24: "b8[32, 512, 768]", mul_122: "f32[32, 512, 768]", view_176: "f32[16384, 768]", where_17: "f32[32, 12, 512, 512]", gt_25: "b8[32, 12, 512, 512]", view_192: "f32[16384, 768]", gt_26: "b8[32, 512, 768]", mul_130: "f32[32, 512, 768]", view_194: "f32[16384, 768]", addmm_52: "f32[16384, 3072]", view_196: "f32[16384, 3072]", gt_27: "b8[32, 512, 768]", mul_137: "f32[32, 512, 768]", view_198: "f32[16384, 768]", where_19: "f32[32, 12, 512, 512]", gt_28: "b8[32, 12, 512, 512]", view_214: "f32[16384, 768]", gt_29: "b8[32, 512, 768]", mul_145: "f32[32, 512, 768]", view_216: "f32[16384, 768]", addmm_58: "f32[16384, 3072]", view_218: "f32[16384, 3072]", gt_30: "b8[32, 512, 768]", mul_152: "f32[32, 512, 768]", view_220: "f32[16384, 768]", where_21: "f32[32, 12, 512, 512]", gt_31: "b8[32, 12, 512, 512]", view_236: "f32[16384, 768]", gt_32: "b8[32, 512, 768]", mul_160: "f32[32, 512, 768]", view_238: "f32[16384, 768]", addmm_64: "f32[16384, 3072]", view_240: "f32[16384, 3072]", gt_33: "b8[32, 512, 768]", mul_167: "f32[32, 512, 768]", view_242: "f32[16384, 768]", where_23: "f32[32, 12, 512, 512]", gt_34: "b8[32, 12, 512, 512]", view_258: "f32[16384, 768]", gt_35: "b8[32, 512, 768]", mul_175: "f32[32, 512, 768]", view_260: "f32[16384, 768]", addmm_70: "f32[16384, 3072]", view_262: "f32[16384, 3072]", gt_36: "b8[32, 512, 768]", mul_182: "f32[32, 512, 768]", view_264: "f32[16384, 768]", addmm_72: "f32[16384, 768]", getitem_51: "f32[32, 512, 1]", rsqrt_25: "f32[32, 512, 1]", view_266: "f32[16384, 768]", view_267: "f32[32, 512, 30522]", amax_12: "f32[16384, 1]", log: "f32[16384, 1]", convert_element_type: "f32[]", div_15: "f32[32, 512, 1]", div_16: "f32[32, 512, 1]", permute_155: "f32[384, 512, 512]", permute_156: "f32[384, 64, 512]", permute_157: "f32[384, 64, 512]", permute_158: "f32[384, 512, 64]", div_17: "f32[32, 512, 1]", div_18: "f32[32, 512, 1]", permute_188: "f32[384, 512, 512]", permute_189: "f32[384, 64, 512]", permute_190: "f32[384, 64, 512]", permute_191: "f32[384, 512, 64]", div_19: "f32[32, 512, 1]", div_20: "f32[32, 512, 1]", permute_221: "f32[384, 512, 512]", permute_222: "f32[384, 64, 512]", permute_223: "f32[384, 64, 512]", permute_224: "f32[384, 512, 64]", div_21: "f32[32, 512, 1]", div_22: "f32[32, 512, 1]", permute_254: "f32[384, 512, 512]", permute_255: "f32[384, 64, 512]", permute_256: "f32[384, 64, 512]", permute_257: "f32[384, 512, 64]", div_23: "f32[32, 512, 1]", div_24: "f32[32, 512, 1]", permute_287: "f32[384, 512, 512]", permute_288: "f32[384, 64, 512]", permute_289: "f32[384, 64, 512]", permute_290: "f32[384, 512, 64]", div_25: "f32[32, 512, 1]", div_26: "f32[32, 512, 1]", permute_320: "f32[384, 512, 512]", permute_321: "f32[384, 64, 512]", permute_322: "f32[384, 64, 512]", permute_323: "f32[384, 512, 64]", div_27: "f32[32, 512, 1]", div_28: "f32[32, 512, 1]", permute_353: "f32[384, 512, 512]", permute_354: "f32[384, 64, 512]", permute_355: "f32[384, 64, 512]", permute_356: "f32[384, 512, 64]", div_29: "f32[32, 512, 1]", div_30: "f32[32, 512, 1]", permute_386: "f32[384, 512, 512]", permute_387: "f32[384, 64, 512]", permute_388: "f32[384, 64, 512]", permute_389: "f32[384, 512, 64]", div_31: "f32[32, 512, 1]", div_32: "f32[32, 512, 1]", permute_419: "f32[384, 512, 512]", permute_420: "f32[384, 64, 512]", permute_421: "f32[384, 64, 512]", permute_422: "f32[384, 512, 64]", div_33: "f32[32, 512, 1]", div_34: "f32[32, 512, 1]", permute_452: "f32[384, 512, 512]", permute_453: "f32[384, 64, 512]", permute_454: "f32[384, 64, 512]", permute_455: "f32[384, 512, 64]", div_35: "f32[32, 512, 1]", div_36: "f32[32, 512, 1]", permute_485: "f32[384, 512, 512]", permute_486: "f32[384, 64, 512]", permute_487: "f32[384, 64, 512]", permute_488: "f32[384, 512, 64]", div_37: "f32[32, 512, 1]", div_38: "f32[32, 512, 1]", permute_518: "f32[384, 512, 512]", permute_519: "f32[384, 64, 512]", permute_520: "f32[384, 64, 512]", permute_521: "f32[384, 512, 64]", div_39: "f32[32, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[32, 512, 30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_269: "i64[16384]" = torch.ops.aten.reshape.default(primals_206, [-1]);  primals_206 = None
        unsqueeze_4: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(view_269, 1);  view_269 = None
        ne_3: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[16384, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_36);  unsqueeze_4 = full_default_36 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[16384, 30522]" = torch.ops.aten.expand.default(where_26, [16384, 30522]);  where_26 = None
        eq_tensor: "b8[16384, 30522]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[16384, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_27: "f32[16384, 1]" = torch.ops.aten.where.self(ne_3, div_13, full_default_1);  ne_3 = div_13 = None
        mul_189: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None
        view_268: "f32[16384, 30522]" = torch.ops.aten.reshape.default(view_267, [-1, 30522]);  view_267 = None
        sub_38: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
        sub_39: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        exp_13: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_16: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [1], True)
        mul_190: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_40: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
        view_270: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(sub_40, [32, 512, 30522]);  sub_40 = None
        add_105: "f32[32, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_270);  tangents_2 = view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        view_271: "f32[16384, 30522]" = torch.ops.aten.reshape.default(add_105, [16384, 30522]);  add_105 = None
        permute_133: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_134: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        constant_pad_nd_default_1: "f32[16384, 30524]" = torch.ops.aten.constant_pad_nd.default(view_271, [0, 2, 0, 0])
        constant_pad_nd_default_2: "f32[30524, 768]" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 0, 0, 2]);  permute_134 = None
        mm_default_1: "f32[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        permute_135: "f32[30522, 16384]" = torch.ops.aten.permute.default(view_271, [1, 0])
        constant_pad_nd_default: "f32[30524, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_135, [0, 0, 0, 2]);  permute_135 = None
        mm_default: "f32[30524, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default, view_266);  constant_pad_nd_default = view_266 = None
        slice_tensor: "f32[30522, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None
        sum_17: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        view_272: "f32[30522]" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
        view_273: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_default_1, [32, 512, 768]);  mm_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_192: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_203);  primals_203 = None
        mul_193: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_18: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_265: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 768]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_184: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
        mul_185: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_186, getitem_51);  mul_186 = getitem_51 = None
        mul_187: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_194: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_187);  mul_192 = None
        sum_19: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, sum_19);  sum_19 = None
        sub_42: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_18);  mul_193 = sum_18 = None
        sub_43: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_195);  sub_42 = mul_195 = None
        div_14: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_196: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_43);  div_14 = sub_43 = None
        mul_197: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_187);  mul_187 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_199: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
        mul_200: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, view_265)
        mul_201: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
        exp_14: "f32[32, 512, 768]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
        mul_202: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_203: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_202);  view_265 = mul_202 = None
        add_107: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
        mul_204: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, add_107);  mul_196 = add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_274: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_204, [16384, 768]);  mul_204 = None
        permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_2: "f32[16384, 768]" = torch.ops.aten.mm.default(view_274, permute_138);  permute_138 = None
        permute_139: "f32[768, 16384]" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
        sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        view_275: "f32[768]" = torch.ops.aten.reshape.default(sum_22, [768]);  sum_22 = None
        view_276: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [32, 512, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_206: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, primals_199);  primals_199 = None
        mul_207: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_23: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_182);  mul_206 = None
        sum_24: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_182, sum_24);  sum_24 = None
        sub_45: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_23);  mul_207 = sum_23 = None
        sub_46: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_209);  sub_45 = mul_209 = None
        mul_210: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
        mul_211: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, mul_182);  mul_182 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 1]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_212: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_213: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_210, mul_212);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_277: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_213, [16384, 768]);  mul_213 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
        permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_4: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_277, permute_142);  permute_142 = None
        permute_143: "f32[768, 16384]" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
        view_278: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_279: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [32, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_178: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_215: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
        mul_216: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_217: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_216, -0.5);  mul_216 = None
        exp_15: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_217);  mul_217 = None
        mul_218: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_219: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_218);  view_261 = mul_218 = None
        add_109: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_215, mul_219);  mul_215 = mul_219 = None
        mul_220: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_279, add_109);  view_279 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_220, [16384, 3072]);  mul_220 = None
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_6: "f32[16384, 768]" = torch.ops.aten.mm.default(view_280, permute_146);  permute_146 = None
        permute_147: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
        sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        view_281: "f32[3072]" = torch.ops.aten.reshape.default(sum_28, [3072]);  sum_28 = None
        view_282: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [32, 512, 768]);  mm_6 = None
        add_110: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_210, view_282);  mul_210 = view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_222: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_193);  primals_193 = None
        mul_223: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
        sum_29: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_175);  mul_222 = None
        sum_30: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_175, sum_30);  sum_30 = None
        sub_48: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_29);  mul_223 = sum_29 = None
        sub_49: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_225);  sub_48 = mul_225 = None
        mul_226: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
        mul_227: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_175);  mul_175 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_228: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_229: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_228);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_229, [16384, 768]);  mul_229 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
        permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_8: "f32[16384, 768]" = torch.ops.aten.mm.default(view_283, permute_150);  permute_150 = None
        permute_151: "f32[768, 16384]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        view_284: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        view_285: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_8, [32, 512, 768]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_286: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_285, [32, 512, 12, 64]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_154: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_50: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_287: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_50, [384, 512, 64]);  clone_50 = None
        bmm_24: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_155, view_287);  permute_155 = None
        bmm_25: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_287, permute_156);  view_287 = permute_156 = None
        view_288: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_24, [32, 12, 512, 64]);  bmm_24 = None
        view_289: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_25, [32, 12, 512, 512]);  bmm_25 = None
        convert_element_type_3: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_230: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_231: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_289, mul_230);  view_289 = mul_230 = None
        mul_232: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_231, where_23);  mul_231 = None
        sum_34: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [-1], True)
        neg_1: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_23);  where_23 = None
        fma: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_232);  neg_1 = sum_34 = mul_232 = None
        view_290: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma, [384, 512, 512]);  fma = None
        bmm_26: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_157, view_290);  permute_157 = None
        bmm_27: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_290, permute_158);  view_290 = permute_158 = None
        view_291: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_26, [32, 12, 64, 512]);  bmm_26 = None
        view_292: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [32, 12, 512, 64]);  bmm_27 = None
        mul_233: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_291, 0.3535533905932738);  view_291 = None
        permute_159: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_233, [0, 1, 3, 2]);  mul_233 = None
        mul_234: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_292, 0.3535533905932738);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_160: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_52: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_293: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_52, [32, 512, 768]);  clone_52 = None
        view_294: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_293, [16384, 768]);  view_293 = None
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_10: "f32[16384, 768]" = torch.ops.aten.mm.default(view_294, permute_161);  permute_161 = None
        permute_162: "f32[768, 16384]" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_242);  permute_162 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        view_295: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        view_296: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [32, 512, 768]);  mm_10 = None
        add_111: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_226, view_296);  mul_226 = view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_165: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        view_297: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_165, [32, 512, 768]);  permute_165 = None
        clone_53: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_297, memory_format = torch.contiguous_format);  view_297 = None
        view_298: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_53, [16384, 768]);  clone_53 = None
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_12: "f32[16384, 768]" = torch.ops.aten.mm.default(view_298, permute_166);  permute_166 = None
        permute_167: "f32[768, 16384]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        view_299: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        view_300: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_12, [32, 512, 768]);  mm_12 = None
        add_112: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_111, view_300);  add_111 = view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_170: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_234, [0, 2, 1, 3]);  mul_234 = None
        clone_54: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_301: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_54, [32, 512, 768]);  clone_54 = None
        view_302: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_301, [16384, 768]);  view_301 = None
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_14: "f32[16384, 768]" = torch.ops.aten.mm.default(view_302, permute_171);  permute_171 = None
        permute_172: "f32[768, 16384]" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
        sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
        view_304: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_14, [32, 512, 768]);  mm_14 = None
        add_113: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_304);  add_112 = view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_236: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_183);  primals_183 = None
        mul_237: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, 768)
        sum_38: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, mul_167);  mul_236 = None
        sum_39: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, sum_39);  sum_39 = None
        sub_51: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_237, sum_38);  mul_237 = sum_38 = None
        sub_52: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_239);  sub_51 = mul_239 = None
        mul_240: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_52);  div_17 = sub_52 = None
        mul_241: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_167);  mul_167 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_242: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_243: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_240, mul_242);  mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_243, [16384, 768]);  mul_243 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_16: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_305, permute_175);  permute_175 = None
        permute_176: "f32[768, 16384]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        view_306: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        view_307: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [32, 512, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_163: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_245: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_246: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_247: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
        exp_16: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
        mul_248: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_249: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_248);  view_239 = mul_248 = None
        add_115: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
        mul_250: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_307, add_115);  view_307 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_308: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_250, [16384, 3072]);  mul_250 = None
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_18: "f32[16384, 768]" = torch.ops.aten.mm.default(view_308, permute_179);  permute_179 = None
        permute_180: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
        sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
        view_309: "f32[3072]" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
        view_310: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [32, 512, 768]);  mm_18 = None
        add_116: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_240, view_310);  mul_240 = view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_252: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_177);  primals_177 = None
        mul_253: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_252, 768)
        sum_44: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_252, mul_160);  mul_252 = None
        sum_45: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_160, sum_45);  sum_45 = None
        sub_54: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_253, sum_44);  mul_253 = sum_44 = None
        sub_55: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_255);  sub_54 = mul_255 = None
        mul_256: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_55);  div_18 = sub_55 = None
        mul_257: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_160);  mul_160 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_5: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_258: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_259: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_256, mul_258);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_311: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_259, [16384, 768]);  mul_259 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_20: "f32[16384, 768]" = torch.ops.aten.mm.default(view_311, permute_183);  permute_183 = None
        permute_184: "f32[768, 16384]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312: "f32[768]" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        view_313: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_20, [32, 512, 768]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_314: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_313, [32, 512, 12, 64]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_187: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_57: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_315: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_57, [384, 512, 64]);  clone_57 = None
        bmm_28: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_188, view_315);  permute_188 = None
        bmm_29: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_315, permute_189);  view_315 = permute_189 = None
        view_316: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_28, [32, 12, 512, 64]);  bmm_28 = None
        view_317: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_29, [32, 12, 512, 512]);  bmm_29 = None
        convert_element_type_6: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_260: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_261: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_317, mul_260);  view_317 = mul_260 = None
        mul_262: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_261, where_21);  mul_261 = None
        sum_49: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_2: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_21);  where_21 = None
        fma_1: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_262);  neg_2 = sum_49 = mul_262 = None
        view_318: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_1, [384, 512, 512]);  fma_1 = None
        bmm_30: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_190, view_318);  permute_190 = None
        bmm_31: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_318, permute_191);  view_318 = permute_191 = None
        view_319: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_30, [32, 12, 64, 512]);  bmm_30 = None
        view_320: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [32, 12, 512, 64]);  bmm_31 = None
        mul_263: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_319, 0.3535533905932738);  view_319 = None
        permute_192: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_263, [0, 1, 3, 2]);  mul_263 = None
        mul_264: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_320, 0.3535533905932738);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_193: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        clone_59: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_321: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_59, [32, 512, 768]);  clone_59 = None
        view_322: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_321, [16384, 768]);  view_321 = None
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_22: "f32[16384, 768]" = torch.ops.aten.mm.default(view_322, permute_194);  permute_194 = None
        permute_195: "f32[768, 16384]" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_220);  permute_195 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
        view_323: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
        view_324: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [32, 512, 768]);  mm_22 = None
        add_117: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_256, view_324);  mul_256 = view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_198: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        view_325: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_198, [32, 512, 768]);  permute_198 = None
        clone_60: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_325, memory_format = torch.contiguous_format);  view_325 = None
        view_326: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_60, [16384, 768]);  clone_60 = None
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_24: "f32[16384, 768]" = torch.ops.aten.mm.default(view_326, permute_199);  permute_199 = None
        permute_200: "f32[768, 16384]" = torch.ops.aten.permute.default(view_326, [1, 0])
        mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
        view_327: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        view_328: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_24, [32, 512, 768]);  mm_24 = None
        add_118: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_117, view_328);  add_117 = view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_203: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_264, [0, 2, 1, 3]);  mul_264 = None
        clone_61: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_329: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_61, [32, 512, 768]);  clone_61 = None
        view_330: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_329, [16384, 768]);  view_329 = None
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_26: "f32[16384, 768]" = torch.ops.aten.mm.default(view_330, permute_204);  permute_204 = None
        permute_205: "f32[768, 16384]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
        sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        view_331: "f32[768]" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
        view_332: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_26, [32, 512, 768]);  mm_26 = None
        add_119: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_118, view_332);  add_118 = view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_266: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_167);  primals_167 = None
        mul_267: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, 768)
        sum_53: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
        mul_268: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, mul_152);  mul_266 = None
        sum_54: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
        mul_269: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_152, sum_54);  sum_54 = None
        sub_57: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_267, sum_53);  mul_267 = sum_53 = None
        sub_58: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_57, mul_269);  sub_57 = mul_269 = None
        mul_270: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_58);  div_19 = sub_58 = None
        mul_271: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, mul_152);  mul_152 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_7: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_272: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_273: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, mul_272);  mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_333: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_273, [16384, 768]);  mul_273 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_28: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_333, permute_208);  permute_208 = None
        permute_209: "f32[768, 16384]" = torch.ops.aten.permute.default(view_333, [1, 0])
        mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
        view_334: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_335: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [32, 512, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_148: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_275: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_276: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_277: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
        exp_17: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
        mul_278: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_279: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_278);  view_217 = mul_278 = None
        add_121: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
        mul_280: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_335, add_121);  view_335 = add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_336: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_280, [16384, 3072]);  mul_280 = None
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_30: "f32[16384, 768]" = torch.ops.aten.mm.default(view_336, permute_212);  permute_212 = None
        permute_213: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
        sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
        view_337: "f32[3072]" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
        view_338: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [32, 512, 768]);  mm_30 = None
        add_122: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_270, view_338);  mul_270 = view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_282: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_161);  primals_161 = None
        mul_283: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, 768)
        sum_59: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
        mul_284: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, mul_145);  mul_282 = None
        sum_60: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        mul_285: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, sum_60);  sum_60 = None
        sub_60: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_283, sum_59);  mul_283 = sum_59 = None
        sub_61: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_285);  sub_60 = mul_285 = None
        mul_286: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_61);  div_20 = sub_61 = None
        mul_287: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_145);  mul_145 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_288: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_289: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_339: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_289, [16384, 768]);  mul_289 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_32: "f32[16384, 768]" = torch.ops.aten.mm.default(view_339, permute_216);  permute_216 = None
        permute_217: "f32[768, 16384]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
        view_340: "f32[768]" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        view_341: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_32, [32, 512, 768]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_342: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_341, [32, 512, 12, 64]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_220: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_64: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_343: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_64, [384, 512, 64]);  clone_64 = None
        bmm_32: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_221, view_343);  permute_221 = None
        bmm_33: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_343, permute_222);  view_343 = permute_222 = None
        view_344: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_32, [32, 12, 512, 64]);  bmm_32 = None
        view_345: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_33, [32, 12, 512, 512]);  bmm_33 = None
        convert_element_type_9: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_290: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_291: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_345, mul_290);  view_345 = mul_290 = None
        mul_292: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_291, where_19);  mul_291 = None
        sum_64: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
        neg_3: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_19);  where_19 = None
        fma_2: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_292);  neg_3 = sum_64 = mul_292 = None
        view_346: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_2, [384, 512, 512]);  fma_2 = None
        bmm_34: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_223, view_346);  permute_223 = None
        bmm_35: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_346, permute_224);  view_346 = permute_224 = None
        view_347: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_34, [32, 12, 64, 512]);  bmm_34 = None
        view_348: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [32, 12, 512, 64]);  bmm_35 = None
        mul_293: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_347, 0.3535533905932738);  view_347 = None
        permute_225: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_293, [0, 1, 3, 2]);  mul_293 = None
        mul_294: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_348, 0.3535533905932738);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_226: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_66: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_349: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_66, [32, 512, 768]);  clone_66 = None
        view_350: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_349, [16384, 768]);  view_349 = None
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_34: "f32[16384, 768]" = torch.ops.aten.mm.default(view_350, permute_227);  permute_227 = None
        permute_228: "f32[768, 16384]" = torch.ops.aten.permute.default(view_350, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_198);  permute_228 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
        view_351: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        view_352: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [32, 512, 768]);  mm_34 = None
        add_123: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_286, view_352);  mul_286 = view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_231: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_353: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_231, [32, 512, 768]);  permute_231 = None
        clone_67: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_353, memory_format = torch.contiguous_format);  view_353 = None
        view_354: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_67, [16384, 768]);  clone_67 = None
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_36: "f32[16384, 768]" = torch.ops.aten.mm.default(view_354, permute_232);  permute_232 = None
        permute_233: "f32[768, 16384]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True);  view_354 = None
        view_355: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        view_356: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_36, [32, 512, 768]);  mm_36 = None
        add_124: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_123, view_356);  add_123 = view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_236: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_294, [0, 2, 1, 3]);  mul_294 = None
        clone_68: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_357: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_68, [32, 512, 768]);  clone_68 = None
        view_358: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_357, [16384, 768]);  view_357 = None
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_38: "f32[16384, 768]" = torch.ops.aten.mm.default(view_358, permute_237);  permute_237 = None
        permute_238: "f32[768, 16384]" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
        sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
        view_359: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
        view_360: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_38, [32, 512, 768]);  mm_38 = None
        add_125: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_360);  add_124 = view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_296: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_151);  primals_151 = None
        mul_297: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_296, 768)
        sum_68: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
        mul_298: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_296, mul_137);  mul_296 = None
        sum_69: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, sum_69);  sum_69 = None
        sub_63: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_297, sum_68);  mul_297 = sum_68 = None
        sub_64: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_299);  sub_63 = mul_299 = None
        mul_300: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_64);  div_21 = sub_64 = None
        mul_301: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_137);  mul_137 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_302: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_303: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_302);  mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_361: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_303, [16384, 768]);  mul_303 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_40: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_361, permute_241);  permute_241 = None
        permute_242: "f32[768, 16384]" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        view_362: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        view_363: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [32, 512, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_133: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_305: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_306: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_307: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
        exp_18: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
        mul_308: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_309: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_308);  view_195 = mul_308 = None
        add_127: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
        mul_310: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_363, add_127);  view_363 = add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_364: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_310, [16384, 3072]);  mul_310 = None
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_42: "f32[16384, 768]" = torch.ops.aten.mm.default(view_364, permute_245);  permute_245 = None
        permute_246: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
        sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
        view_365: "f32[3072]" = torch.ops.aten.reshape.default(sum_73, [3072]);  sum_73 = None
        view_366: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [32, 512, 768]);  mm_42 = None
        add_128: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_300, view_366);  mul_300 = view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_312: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_145);  primals_145 = None
        mul_313: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, 768)
        sum_74: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
        mul_314: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, mul_130);  mul_312 = None
        sum_75: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
        mul_315: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, sum_75);  sum_75 = None
        sub_66: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_313, sum_74);  mul_313 = sum_74 = None
        sub_67: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_315);  sub_66 = mul_315 = None
        mul_316: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_67);  div_22 = sub_67 = None
        mul_317: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_130);  mul_130 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_11: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_318: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_319: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_318);  mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_367: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_319, [16384, 768]);  mul_319 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_44: "f32[16384, 768]" = torch.ops.aten.mm.default(view_367, permute_249);  permute_249 = None
        permute_250: "f32[768, 16384]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        view_368: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        view_369: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_44, [32, 512, 768]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_370: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_369, [32, 512, 12, 64]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_253: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_71: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_371: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_71, [384, 512, 64]);  clone_71 = None
        bmm_36: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_254, view_371);  permute_254 = None
        bmm_37: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_371, permute_255);  view_371 = permute_255 = None
        view_372: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_36, [32, 12, 512, 64]);  bmm_36 = None
        view_373: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_37, [32, 12, 512, 512]);  bmm_37 = None
        convert_element_type_12: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_320: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_321: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_373, mul_320);  view_373 = mul_320 = None
        mul_322: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_321, where_17);  mul_321 = None
        sum_79: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
        neg_4: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_17);  where_17 = None
        fma_3: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_322);  neg_4 = sum_79 = mul_322 = None
        view_374: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_3, [384, 512, 512]);  fma_3 = None
        bmm_38: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_256, view_374);  permute_256 = None
        bmm_39: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_374, permute_257);  view_374 = permute_257 = None
        view_375: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_38, [32, 12, 64, 512]);  bmm_38 = None
        view_376: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [32, 12, 512, 64]);  bmm_39 = None
        mul_323: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_375, 0.3535533905932738);  view_375 = None
        permute_258: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_323, [0, 1, 3, 2]);  mul_323 = None
        mul_324: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_376, 0.3535533905932738);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_259: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        clone_73: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_377: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_73, [32, 512, 768]);  clone_73 = None
        view_378: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_377, [16384, 768]);  view_377 = None
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_46: "f32[16384, 768]" = torch.ops.aten.mm.default(view_378, permute_260);  permute_260 = None
        permute_261: "f32[768, 16384]" = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view_176);  permute_261 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
        view_379: "f32[768]" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
        view_380: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [32, 512, 768]);  mm_46 = None
        add_129: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_316, view_380);  mul_316 = view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_264: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        view_381: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_264, [32, 512, 768]);  permute_264 = None
        clone_74: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_381, memory_format = torch.contiguous_format);  view_381 = None
        view_382: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_74, [16384, 768]);  clone_74 = None
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_48: "f32[16384, 768]" = torch.ops.aten.mm.default(view_382, permute_265);  permute_265 = None
        permute_266: "f32[768, 16384]" = torch.ops.aten.permute.default(view_382, [1, 0])
        mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
        view_383: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        view_384: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_48, [32, 512, 768]);  mm_48 = None
        add_130: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_384);  add_129 = view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_269: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_324, [0, 2, 1, 3]);  mul_324 = None
        clone_75: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_385: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_75, [32, 512, 768]);  clone_75 = None
        view_386: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_385, [16384, 768]);  view_385 = None
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_50: "f32[16384, 768]" = torch.ops.aten.mm.default(view_386, permute_270);  permute_270 = None
        permute_271: "f32[768, 16384]" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
        sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
        view_387: "f32[768]" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
        view_388: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_50, [32, 512, 768]);  mm_50 = None
        add_131: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_130, view_388);  add_130 = view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_326: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_135);  primals_135 = None
        mul_327: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_326, 768)
        sum_83: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_326, mul_122);  mul_326 = None
        sum_84: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, sum_84);  sum_84 = None
        sub_69: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_327, sum_83);  mul_327 = sum_83 = None
        sub_70: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_329);  sub_69 = mul_329 = None
        mul_330: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_70);  div_23 = sub_70 = None
        mul_331: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_122);  mul_122 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_332: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_333: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_330, mul_332);  mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_389: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_333, [16384, 768]);  mul_333 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_274: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_52: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_389, permute_274);  permute_274 = None
        permute_275: "f32[768, 16384]" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        view_390: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_391: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_52, [32, 512, 3072]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_335: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_336: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_337: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_336, -0.5);  mul_336 = None
        exp_19: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_337);  mul_337 = None
        mul_338: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_339: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_338);  view_173 = mul_338 = None
        add_133: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_335, mul_339);  mul_335 = mul_339 = None
        mul_340: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_391, add_133);  view_391 = add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_340, [16384, 3072]);  mul_340 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_54: "f32[16384, 768]" = torch.ops.aten.mm.default(view_392, permute_278);  permute_278 = None
        permute_279: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
        sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
        view_393: "f32[3072]" = torch.ops.aten.reshape.default(sum_88, [3072]);  sum_88 = None
        view_394: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_54, [32, 512, 768]);  mm_54 = None
        add_134: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_330, view_394);  mul_330 = view_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_342: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_129);  primals_129 = None
        mul_343: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
        sum_89: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_115);  mul_342 = None
        sum_90: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, sum_90);  sum_90 = None
        sub_72: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_89);  mul_343 = sum_89 = None
        sub_73: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_345);  sub_72 = mul_345 = None
        mul_346: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_73);  div_24 = sub_73 = None
        mul_347: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_115);  mul_115 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_348: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_349: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_346, mul_348);  mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_395: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_349, [16384, 768]);  mul_349 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_56: "f32[16384, 768]" = torch.ops.aten.mm.default(view_395, permute_282);  permute_282 = None
        permute_283: "f32[768, 16384]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        view_396: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        view_397: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_56, [32, 512, 768]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_398: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_397, [32, 512, 12, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_286: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_78: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_399: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_78, [384, 512, 64]);  clone_78 = None
        bmm_40: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_399);  permute_287 = None
        bmm_41: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_399, permute_288);  view_399 = permute_288 = None
        view_400: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_40, [32, 12, 512, 64]);  bmm_40 = None
        view_401: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_41, [32, 12, 512, 512]);  bmm_41 = None
        convert_element_type_15: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_350: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_351: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_401, mul_350);  view_401 = mul_350 = None
        mul_352: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_351, where_15);  mul_351 = None
        sum_94: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_5: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_15);  where_15 = None
        fma_4: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_352);  neg_5 = sum_94 = mul_352 = None
        view_402: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_4, [384, 512, 512]);  fma_4 = None
        bmm_42: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_402);  permute_289 = None
        bmm_43: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_402, permute_290);  view_402 = permute_290 = None
        view_403: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_42, [32, 12, 64, 512]);  bmm_42 = None
        view_404: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [32, 12, 512, 64]);  bmm_43 = None
        mul_353: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_403, 0.3535533905932738);  view_403 = None
        permute_291: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_353, [0, 1, 3, 2]);  mul_353 = None
        mul_354: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_404, 0.3535533905932738);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_292: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_80: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_405: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_80, [32, 512, 768]);  clone_80 = None
        view_406: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_405, [16384, 768]);  view_405 = None
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_293: "f32[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_58: "f32[16384, 768]" = torch.ops.aten.mm.default(view_406, permute_293);  permute_293 = None
        permute_294: "f32[768, 16384]" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_294, view_154);  permute_294 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        view_407: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
        view_408: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_58, [32, 512, 768]);  mm_58 = None
        add_135: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_346, view_408);  mul_346 = view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_297: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
        view_409: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_297, [32, 512, 768]);  permute_297 = None
        clone_81: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_409, memory_format = torch.contiguous_format);  view_409 = None
        view_410: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_81, [16384, 768]);  clone_81 = None
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_60: "f32[16384, 768]" = torch.ops.aten.mm.default(view_410, permute_298);  permute_298 = None
        permute_299: "f32[768, 16384]" = torch.ops.aten.permute.default(view_410, [1, 0])
        mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_410, [0], True);  view_410 = None
        view_411: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        view_412: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_60, [32, 512, 768]);  mm_60 = None
        add_136: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_135, view_412);  add_135 = view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_302: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_354, [0, 2, 1, 3]);  mul_354 = None
        clone_82: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_413: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_82, [32, 512, 768]);  clone_82 = None
        view_414: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_413, [16384, 768]);  view_413 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_62: "f32[16384, 768]" = torch.ops.aten.mm.default(view_414, permute_303);  permute_303 = None
        permute_304: "f32[768, 16384]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
        sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
        view_415: "f32[768]" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        view_416: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_62, [32, 512, 768]);  mm_62 = None
        add_137: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_416);  add_136 = view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_356: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_119);  primals_119 = None
        mul_357: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, 768)
        sum_98: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, mul_107);  mul_356 = None
        sum_99: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, sum_99);  sum_99 = None
        sub_75: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_357, sum_98);  mul_357 = sum_98 = None
        sub_76: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_359);  sub_75 = mul_359 = None
        mul_360: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_76);  div_25 = sub_76 = None
        mul_361: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_107);  mul_107 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_362: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_363: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_360, mul_362);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_417: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_363, [16384, 768]);  mul_363 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_64: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_417, permute_307);  permute_307 = None
        permute_308: "f32[768, 16384]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
        view_418: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        view_419: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_64, [32, 512, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_103: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_365: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
        mul_366: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_367: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_366, -0.5);  mul_366 = None
        exp_20: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_367);  mul_367 = None
        mul_368: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_369: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_368);  view_151 = mul_368 = None
        add_139: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_365, mul_369);  mul_365 = mul_369 = None
        mul_370: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_419, add_139);  view_419 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_420: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_370, [16384, 3072]);  mul_370 = None
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_66: "f32[16384, 768]" = torch.ops.aten.mm.default(view_420, permute_311);  permute_311 = None
        permute_312: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
        sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
        view_421: "f32[3072]" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
        view_422: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_66, [32, 512, 768]);  mm_66 = None
        add_140: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_360, view_422);  mul_360 = view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_372: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_113);  primals_113 = None
        mul_373: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, 768)
        sum_104: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
        mul_374: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, mul_100);  mul_372 = None
        sum_105: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
        mul_375: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_105);  sum_105 = None
        sub_78: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_373, sum_104);  mul_373 = sum_104 = None
        sub_79: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_375);  sub_78 = mul_375 = None
        mul_376: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_79);  div_26 = sub_79 = None
        mul_377: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, mul_100);  mul_100 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
        sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_378: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_379: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_376, mul_378);  mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_423: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_379, [16384, 768]);  mul_379 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_68: "f32[16384, 768]" = torch.ops.aten.mm.default(view_423, permute_315);  permute_315 = None
        permute_316: "f32[768, 16384]" = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
        sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
        view_424: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        view_425: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_68, [32, 512, 768]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_426: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_425, [32, 512, 12, 64]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_319: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_85: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_427: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_85, [384, 512, 64]);  clone_85 = None
        bmm_44: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_427);  permute_320 = None
        bmm_45: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_427, permute_321);  view_427 = permute_321 = None
        view_428: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_44, [32, 12, 512, 64]);  bmm_44 = None
        view_429: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_45, [32, 12, 512, 512]);  bmm_45 = None
        convert_element_type_18: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_380: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_381: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_429, mul_380);  view_429 = mul_380 = None
        mul_382: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_381, where_13);  mul_381 = None
        sum_109: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [-1], True)
        neg_6: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_13);  where_13 = None
        fma_5: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_382);  neg_6 = sum_109 = mul_382 = None
        view_430: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_5, [384, 512, 512]);  fma_5 = None
        bmm_46: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_322, view_430);  permute_322 = None
        bmm_47: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_430, permute_323);  view_430 = permute_323 = None
        view_431: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_46, [32, 12, 64, 512]);  bmm_46 = None
        view_432: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [32, 12, 512, 64]);  bmm_47 = None
        mul_383: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_431, 0.3535533905932738);  view_431 = None
        permute_324: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_383, [0, 1, 3, 2]);  mul_383 = None
        mul_384: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_432, 0.3535533905932738);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_325: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_87: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_433: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_87, [32, 512, 768]);  clone_87 = None
        view_434: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_433, [16384, 768]);  view_433 = None
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_326: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_70: "f32[16384, 768]" = torch.ops.aten.mm.default(view_434, permute_326);  permute_326 = None
        permute_327: "f32[768, 16384]" = torch.ops.aten.permute.default(view_434, [1, 0])
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_327, view_132);  permute_327 = None
        sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
        view_435: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
        view_436: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_70, [32, 512, 768]);  mm_70 = None
        add_141: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_376, view_436);  mul_376 = view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_330: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
        view_437: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_330, [32, 512, 768]);  permute_330 = None
        clone_88: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_437, memory_format = torch.contiguous_format);  view_437 = None
        view_438: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_88, [16384, 768]);  clone_88 = None
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_72: "f32[16384, 768]" = torch.ops.aten.mm.default(view_438, permute_331);  permute_331 = None
        permute_332: "f32[768, 16384]" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = None
        sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
        view_439: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        view_440: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_72, [32, 512, 768]);  mm_72 = None
        add_142: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_141, view_440);  add_141 = view_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_335: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_384, [0, 2, 1, 3]);  mul_384 = None
        clone_89: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_441: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_89, [32, 512, 768]);  clone_89 = None
        view_442: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_441, [16384, 768]);  view_441 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_74: "f32[16384, 768]" = torch.ops.aten.mm.default(view_442, permute_336);  permute_336 = None
        permute_337: "f32[768, 16384]" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
        sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
        view_443: "f32[768]" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
        view_444: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_74, [32, 512, 768]);  mm_74 = None
        add_143: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_444);  add_142 = view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_386: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_103);  primals_103 = None
        mul_387: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_113: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, mul_92);  mul_386 = None
        sum_114: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, sum_114);  sum_114 = None
        sub_81: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_387, sum_113);  mul_387 = sum_113 = None
        sub_82: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_81, mul_389);  sub_81 = mul_389 = None
        mul_390: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_82);  div_27 = sub_82 = None
        mul_391: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_92);  mul_92 = None
        sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_392: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_393: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_390, mul_392);  mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_445: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_393, [16384, 768]);  mul_393 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_340: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_76: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_445, permute_340);  permute_340 = None
        permute_341: "f32[768, 16384]" = torch.ops.aten.permute.default(view_445, [1, 0])
        mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
        view_446: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_447: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_76, [32, 512, 3072]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_395: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_396: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_397: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_21: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_399: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_398);  view_129 = mul_398 = None
        add_145: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_145);  view_447 = add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_448: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_400, [16384, 3072]);  mul_400 = None
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_78: "f32[16384, 768]" = torch.ops.aten.mm.default(view_448, permute_344);  permute_344 = None
        permute_345: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
        sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
        view_449: "f32[3072]" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
        view_450: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_78, [32, 512, 768]);  mm_78 = None
        add_146: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_390, view_450);  mul_390 = view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_402: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_97);  primals_97 = None
        mul_403: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, 768)
        sum_119: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_85);  mul_402 = None
        sum_120: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_120);  sum_120 = None
        sub_84: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_403, sum_119);  mul_403 = sum_119 = None
        sub_85: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_405);  sub_84 = mul_405 = None
        mul_406: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_85);  div_28 = sub_85 = None
        mul_407: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_85);  mul_85 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_408: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_409: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_406, mul_408);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_451: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_409, [16384, 768]);  mul_409 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_348: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_80: "f32[16384, 768]" = torch.ops.aten.mm.default(view_451, permute_348);  permute_348 = None
        permute_349: "f32[768, 16384]" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
        view_452: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        view_453: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_80, [32, 512, 768]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_454: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_453, [32, 512, 12, 64]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_352: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_92: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_455: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_92, [384, 512, 64]);  clone_92 = None
        bmm_48: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_353, view_455);  permute_353 = None
        bmm_49: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_455, permute_354);  view_455 = permute_354 = None
        view_456: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [32, 12, 512, 64]);  bmm_48 = None
        view_457: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [32, 12, 512, 512]);  bmm_49 = None
        convert_element_type_21: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_410: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_411: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_457, mul_410);  view_457 = mul_410 = None
        mul_412: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_411, where_11);  mul_411 = None
        sum_124: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_7: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_11);  where_11 = None
        fma_6: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_412);  neg_7 = sum_124 = mul_412 = None
        view_458: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_6, [384, 512, 512]);  fma_6 = None
        bmm_50: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_355, view_458);  permute_355 = None
        bmm_51: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_458, permute_356);  view_458 = permute_356 = None
        view_459: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_50, [32, 12, 64, 512]);  bmm_50 = None
        view_460: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [32, 12, 512, 64]);  bmm_51 = None
        mul_413: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_459, 0.3535533905932738);  view_459 = None
        permute_357: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_413, [0, 1, 3, 2]);  mul_413 = None
        mul_414: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_460, 0.3535533905932738);  view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_358: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_94: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_461: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_94, [32, 512, 768]);  clone_94 = None
        view_462: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_461, [16384, 768]);  view_461 = None
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_359: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_82: "f32[16384, 768]" = torch.ops.aten.mm.default(view_462, permute_359);  permute_359 = None
        permute_360: "f32[768, 16384]" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_360, view_110);  permute_360 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
        view_463: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        view_464: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_82, [32, 512, 768]);  mm_82 = None
        add_147: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_406, view_464);  mul_406 = view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_363: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
        view_465: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_363, [32, 512, 768]);  permute_363 = None
        clone_95: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_465, memory_format = torch.contiguous_format);  view_465 = None
        view_466: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_95, [16384, 768]);  clone_95 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_84: "f32[16384, 768]" = torch.ops.aten.mm.default(view_466, permute_364);  permute_364 = None
        permute_365: "f32[768, 16384]" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = None
        sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
        view_467: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        view_468: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_84, [32, 512, 768]);  mm_84 = None
        add_148: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_147, view_468);  add_147 = view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_368: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_414, [0, 2, 1, 3]);  mul_414 = None
        clone_96: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_469: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_96, [32, 512, 768]);  clone_96 = None
        view_470: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_469, [16384, 768]);  view_469 = None
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_86: "f32[16384, 768]" = torch.ops.aten.mm.default(view_470, permute_369);  permute_369 = None
        permute_370: "f32[768, 16384]" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
        sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        view_471: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
        view_472: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_86, [32, 512, 768]);  mm_86 = None
        add_149: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_148, view_472);  add_148 = view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_416: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_87);  primals_87 = None
        mul_417: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, 768)
        sum_128: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
        mul_418: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_77);  mul_416 = None
        sum_129: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_129);  sum_129 = None
        sub_87: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_417, sum_128);  mul_417 = sum_128 = None
        sub_88: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_419);  sub_87 = mul_419 = None
        mul_420: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_88);  div_29 = sub_88 = None
        mul_421: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_77);  mul_77 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
        sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_422: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_423: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_420, mul_422);  mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_473: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_423, [16384, 768]);  mul_423 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_373: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_88: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_473, permute_373);  permute_373 = None
        permute_374: "f32[768, 16384]" = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
        view_474: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        view_475: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_88, [32, 512, 3072]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_73: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_425: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_426: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_427: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_426, -0.5);  mul_426 = None
        exp_22: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_427);  mul_427 = None
        mul_428: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_429: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_428);  view_107 = mul_428 = None
        add_151: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_425, mul_429);  mul_425 = mul_429 = None
        mul_430: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_475, add_151);  view_475 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_476: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_430, [16384, 3072]);  mul_430 = None
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_90: "f32[16384, 768]" = torch.ops.aten.mm.default(view_476, permute_377);  permute_377 = None
        permute_378: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_476, [1, 0])
        mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
        sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
        view_477: "f32[3072]" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
        view_478: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_90, [32, 512, 768]);  mm_90 = None
        add_152: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_420, view_478);  mul_420 = view_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_432: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_81);  primals_81 = None
        mul_433: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_432, 768)
        sum_134: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_432, mul_70);  mul_432 = None
        sum_135: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_135);  sum_135 = None
        sub_90: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_433, sum_134);  mul_433 = sum_134 = None
        sub_91: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_435);  sub_90 = mul_435 = None
        mul_436: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_91);  div_30 = sub_91 = None
        mul_437: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, mul_70);  mul_70 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_438: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_439: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_436, mul_438);  mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_479: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_439, [16384, 768]);  mul_439 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_92: "f32[16384, 768]" = torch.ops.aten.mm.default(view_479, permute_381);  permute_381 = None
        permute_382: "f32[768, 16384]" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
        view_480: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        view_481: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_92, [32, 512, 768]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_482: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_481, [32, 512, 12, 64]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_385: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_99: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_483: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_99, [384, 512, 64]);  clone_99 = None
        bmm_52: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_386, view_483);  permute_386 = None
        bmm_53: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_483, permute_387);  view_483 = permute_387 = None
        view_484: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [32, 12, 512, 64]);  bmm_52 = None
        view_485: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_53, [32, 12, 512, 512]);  bmm_53 = None
        convert_element_type_24: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_440: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_441: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_485, mul_440);  view_485 = mul_440 = None
        mul_442: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_441, where_9);  mul_441 = None
        sum_139: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_8: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_9);  where_9 = None
        fma_7: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_442);  neg_8 = sum_139 = mul_442 = None
        view_486: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_7, [384, 512, 512]);  fma_7 = None
        bmm_54: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_388, view_486);  permute_388 = None
        bmm_55: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_486, permute_389);  view_486 = permute_389 = None
        view_487: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_54, [32, 12, 64, 512]);  bmm_54 = None
        view_488: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [32, 12, 512, 64]);  bmm_55 = None
        mul_443: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_487, 0.3535533905932738);  view_487 = None
        permute_390: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_443, [0, 1, 3, 2]);  mul_443 = None
        mul_444: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_488, 0.3535533905932738);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_391: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
        clone_101: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_489: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_101, [32, 512, 768]);  clone_101 = None
        view_490: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_489, [16384, 768]);  view_489 = None
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_392: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_94: "f32[16384, 768]" = torch.ops.aten.mm.default(view_490, permute_392);  permute_392 = None
        permute_393: "f32[768, 16384]" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_393, view_88);  permute_393 = None
        sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
        view_491: "f32[768]" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
        view_492: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_94, [32, 512, 768]);  mm_94 = None
        add_153: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_436, view_492);  mul_436 = view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_396: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
        view_493: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_396, [32, 512, 768]);  permute_396 = None
        clone_102: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_493, memory_format = torch.contiguous_format);  view_493 = None
        view_494: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_102, [16384, 768]);  clone_102 = None
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_96: "f32[16384, 768]" = torch.ops.aten.mm.default(view_494, permute_397);  permute_397 = None
        permute_398: "f32[768, 16384]" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = None
        sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
        view_495: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        view_496: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_96, [32, 512, 768]);  mm_96 = None
        add_154: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_153, view_496);  add_153 = view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_401: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_444, [0, 2, 1, 3]);  mul_444 = None
        clone_103: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_497: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_103, [32, 512, 768]);  clone_103 = None
        view_498: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_497, [16384, 768]);  view_497 = None
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_98: "f32[16384, 768]" = torch.ops.aten.mm.default(view_498, permute_402);  permute_402 = None
        permute_403: "f32[768, 16384]" = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
        sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
        view_499: "f32[768]" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None
        view_500: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_98, [32, 512, 768]);  mm_98 = None
        add_155: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_500);  add_154 = view_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_446: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_71);  primals_71 = None
        mul_447: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_446, 768)
        sum_143: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        mul_448: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_446, mul_62);  mul_446 = None
        sum_144: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, sum_144);  sum_144 = None
        sub_93: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_447, sum_143);  mul_447 = sum_143 = None
        sub_94: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_93, mul_449);  sub_93 = mul_449 = None
        mul_450: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_94);  div_31 = sub_94 = None
        mul_451: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_62);  mul_62 = None
        sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_452: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_453: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_450, mul_452);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_453, [16384, 768]);  mul_453 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_100: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_501, permute_406);  permute_406 = None
        permute_407: "f32[768, 16384]" = torch.ops.aten.permute.default(view_501, [1, 0])
        mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
        view_502: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_503: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_100, [32, 512, 3072]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_455: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_456: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_457: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_456, -0.5);  mul_456 = None
        exp_23: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_457);  mul_457 = None
        mul_458: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_459: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_458);  view_85 = mul_458 = None
        add_157: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_455, mul_459);  mul_455 = mul_459 = None
        mul_460: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, add_157);  view_503 = add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_460, [16384, 3072]);  mul_460 = None
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_410: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_102: "f32[16384, 768]" = torch.ops.aten.mm.default(view_504, permute_410);  permute_410 = None
        permute_411: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
        sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
        view_505: "f32[3072]" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
        view_506: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_102, [32, 512, 768]);  mm_102 = None
        add_158: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_450, view_506);  mul_450 = view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_462: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_65);  primals_65 = None
        mul_463: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_462, 768)
        sum_149: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_462, mul_55);  mul_462 = None
        sum_150: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_55, sum_150);  sum_150 = None
        sub_96: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_463, sum_149);  mul_463 = sum_149 = None
        sub_97: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_465);  sub_96 = mul_465 = None
        mul_466: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_97);  div_32 = sub_97 = None
        mul_467: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, mul_55);  mul_55 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_468: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_469: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_466, mul_468);  mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_507: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_469, [16384, 768]);  mul_469 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_104: "f32[16384, 768]" = torch.ops.aten.mm.default(view_507, permute_414);  permute_414 = None
        permute_415: "f32[768, 16384]" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
        sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
        view_508: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        view_509: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_104, [32, 512, 768]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_510: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_509, [32, 512, 12, 64]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_418: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_106: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_511: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_106, [384, 512, 64]);  clone_106 = None
        bmm_56: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_419, view_511);  permute_419 = None
        bmm_57: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_511, permute_420);  view_511 = permute_420 = None
        view_512: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [32, 12, 512, 64]);  bmm_56 = None
        view_513: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_57, [32, 12, 512, 512]);  bmm_57 = None
        convert_element_type_27: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_470: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_471: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_513, mul_470);  view_513 = mul_470 = None
        mul_472: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_471, where_7);  mul_471 = None
        sum_154: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
        neg_9: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_7);  where_7 = None
        fma_8: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_472);  neg_9 = sum_154 = mul_472 = None
        view_514: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_8, [384, 512, 512]);  fma_8 = None
        bmm_58: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_421, view_514);  permute_421 = None
        bmm_59: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_514, permute_422);  view_514 = permute_422 = None
        view_515: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_58, [32, 12, 64, 512]);  bmm_58 = None
        view_516: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [32, 12, 512, 64]);  bmm_59 = None
        mul_473: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_515, 0.3535533905932738);  view_515 = None
        permute_423: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_473, [0, 1, 3, 2]);  mul_473 = None
        mul_474: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_516, 0.3535533905932738);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_424: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_108: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_517: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_108, [32, 512, 768]);  clone_108 = None
        view_518: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_517, [16384, 768]);  view_517 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_425: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_106: "f32[16384, 768]" = torch.ops.aten.mm.default(view_518, permute_425);  permute_425 = None
        permute_426: "f32[768, 16384]" = torch.ops.aten.permute.default(view_518, [1, 0])
        mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_426, view_66);  permute_426 = None
        sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
        view_519: "f32[768]" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
        view_520: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_106, [32, 512, 768]);  mm_106 = None
        add_159: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_466, view_520);  mul_466 = view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_429: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
        view_521: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_429, [32, 512, 768]);  permute_429 = None
        clone_109: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_521, memory_format = torch.contiguous_format);  view_521 = None
        view_522: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_109, [16384, 768]);  clone_109 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_108: "f32[16384, 768]" = torch.ops.aten.mm.default(view_522, permute_430);  permute_430 = None
        permute_431: "f32[768, 16384]" = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = None
        sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
        view_523: "f32[768]" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
        view_524: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_108, [32, 512, 768]);  mm_108 = None
        add_160: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_159, view_524);  add_159 = view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_434: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_474, [0, 2, 1, 3]);  mul_474 = None
        clone_110: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_525: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_110, [32, 512, 768]);  clone_110 = None
        view_526: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_525, [16384, 768]);  view_525 = None
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_110: "f32[16384, 768]" = torch.ops.aten.mm.default(view_526, permute_435);  permute_435 = None
        permute_436: "f32[768, 16384]" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
        sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
        view_527: "f32[768]" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None
        view_528: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_110, [32, 512, 768]);  mm_110 = None
        add_161: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_160, view_528);  add_160 = view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, primals_55);  primals_55 = None
        mul_477: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_476, 768)
        sum_158: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_476, mul_47);  mul_476 = None
        sum_159: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_47, sum_159);  sum_159 = None
        sub_99: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_477, sum_158);  mul_477 = sum_158 = None
        sub_100: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_99, mul_479);  sub_99 = mul_479 = None
        mul_480: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_100);  div_33 = sub_100 = None
        mul_481: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_47);  mul_47 = None
        sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_482: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_483: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_480, mul_482);  mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_483, [16384, 768]);  mul_483 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_112: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_529, permute_439);  permute_439 = None
        permute_440: "f32[768, 16384]" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
        sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        view_530: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        view_531: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_112, [32, 512, 3072]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_43: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_485: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_486: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_487: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_24: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_489: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_488);  view_63 = mul_488 = None
        add_163: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_531, add_163);  view_531 = add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_532: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_490, [16384, 3072]);  mul_490 = None
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_443: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_114: "f32[16384, 768]" = torch.ops.aten.mm.default(view_532, permute_443);  permute_443 = None
        permute_444: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
        sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
        view_533: "f32[3072]" = torch.ops.aten.reshape.default(sum_163, [3072]);  sum_163 = None
        view_534: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_114, [32, 512, 768]);  mm_114 = None
        add_164: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_480, view_534);  mul_480 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_49);  primals_49 = None
        mul_493: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_492, 768)
        sum_164: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_492, mul_40);  mul_492 = None
        sum_165: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_165);  sum_165 = None
        sub_102: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_493, sum_164);  mul_493 = sum_164 = None
        sub_103: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_495);  sub_102 = mul_495 = None
        mul_496: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_103);  div_34 = sub_103 = None
        mul_497: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_40);  mul_40 = None
        sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_498: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_499: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_496, mul_498);  mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_535: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_499, [16384, 768]);  mul_499 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_447: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_116: "f32[16384, 768]" = torch.ops.aten.mm.default(view_535, permute_447);  permute_447 = None
        permute_448: "f32[768, 16384]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
        sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[768]" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
        view_537: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_116, [32, 512, 768]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_538: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_537, [32, 512, 12, 64]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_451: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_113: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_539: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_113, [384, 512, 64]);  clone_113 = None
        bmm_60: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_452, view_539);  permute_452 = None
        bmm_61: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_539, permute_453);  view_539 = permute_453 = None
        view_540: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [32, 12, 512, 64]);  bmm_60 = None
        view_541: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_61, [32, 12, 512, 512]);  bmm_61 = None
        convert_element_type_30: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_500: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_501: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_541, mul_500);  view_541 = mul_500 = None
        mul_502: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_501, where_5);  mul_501 = None
        sum_169: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_10: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_5);  where_5 = None
        fma_9: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_502);  neg_10 = sum_169 = mul_502 = None
        view_542: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_9, [384, 512, 512]);  fma_9 = None
        bmm_62: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_454, view_542);  permute_454 = None
        bmm_63: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_542, permute_455);  view_542 = permute_455 = None
        view_543: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_62, [32, 12, 64, 512]);  bmm_62 = None
        view_544: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [32, 12, 512, 64]);  bmm_63 = None
        mul_503: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_543, 0.3535533905932738);  view_543 = None
        permute_456: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_503, [0, 1, 3, 2]);  mul_503 = None
        mul_504: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_544, 0.3535533905932738);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_457: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        clone_115: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_545: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_115, [32, 512, 768]);  clone_115 = None
        view_546: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_545, [16384, 768]);  view_545 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_458: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_118: "f32[16384, 768]" = torch.ops.aten.mm.default(view_546, permute_458);  permute_458 = None
        permute_459: "f32[768, 16384]" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_459, view_44);  permute_459 = None
        sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
        view_547: "f32[768]" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
        view_548: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_118, [32, 512, 768]);  mm_118 = None
        add_165: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_496, view_548);  mul_496 = view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_462: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
        view_549: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_462, [32, 512, 768]);  permute_462 = None
        clone_116: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_549, memory_format = torch.contiguous_format);  view_549 = None
        view_550: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_116, [16384, 768]);  clone_116 = None
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_120: "f32[16384, 768]" = torch.ops.aten.mm.default(view_550, permute_463);  permute_463 = None
        permute_464: "f32[768, 16384]" = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = None
        sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
        view_551: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
        view_552: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_120, [32, 512, 768]);  mm_120 = None
        add_166: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_165, view_552);  add_165 = view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_467: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_504, [0, 2, 1, 3]);  mul_504 = None
        clone_117: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_553: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_117, [32, 512, 768]);  clone_117 = None
        view_554: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_553, [16384, 768]);  view_553 = None
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_122: "f32[16384, 768]" = torch.ops.aten.mm.default(view_554, permute_468);  permute_468 = None
        permute_469: "f32[768, 16384]" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
        sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
        view_555: "f32[768]" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
        view_556: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_122, [32, 512, 768]);  mm_122 = None
        add_167: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_166, view_556);  add_166 = view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_506: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_39);  primals_39 = None
        mul_507: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_506, 768)
        sum_173: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
        mul_508: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_506, mul_32);  mul_506 = None
        sum_174: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_174);  sum_174 = None
        sub_105: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_507, sum_173);  mul_507 = sum_173 = None
        sub_106: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_509);  sub_105 = mul_509 = None
        mul_510: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_106);  div_35 = sub_106 = None
        mul_511: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_32);  mul_32 = None
        sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
        sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_512: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_513: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_510, mul_512);  mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_557: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_513, [16384, 768]);  mul_513 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_472: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_124: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_557, permute_472);  permute_472 = None
        permute_473: "f32[768, 16384]" = torch.ops.aten.permute.default(view_557, [1, 0])
        mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
        sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
        view_558: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        view_559: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_124, [32, 512, 3072]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_515: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_516: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_517: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
        exp_25: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
        mul_518: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_519: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_518);  view_41 = mul_518 = None
        add_169: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
        mul_520: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_559, add_169);  view_559 = add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_560: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_520, [16384, 3072]);  mul_520 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_476: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_126: "f32[16384, 768]" = torch.ops.aten.mm.default(view_560, permute_476);  permute_476 = None
        permute_477: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
        sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
        view_561: "f32[3072]" = torch.ops.aten.reshape.default(sum_178, [3072]);  sum_178 = None
        view_562: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_126, [32, 512, 768]);  mm_126 = None
        add_170: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_510, view_562);  mul_510 = view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_522: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_33);  primals_33 = None
        mul_523: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_522, 768)
        sum_179: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
        mul_524: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_522, mul_25);  mul_522 = None
        sum_180: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
        mul_525: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_180);  sum_180 = None
        sub_108: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_523, sum_179);  mul_523 = sum_179 = None
        sub_109: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_525);  sub_108 = mul_525 = None
        mul_526: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_109);  div_36 = sub_109 = None
        mul_527: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, mul_25);  mul_25 = None
        sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
        sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_528: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_529: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_526, mul_528);  mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_563: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_529, [16384, 768]);  mul_529 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_128: "f32[16384, 768]" = torch.ops.aten.mm.default(view_563, permute_480);  permute_480 = None
        permute_481: "f32[768, 16384]" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
        sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        view_564: "f32[768]" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
        view_565: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_128, [32, 512, 768]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_566: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_565, [32, 512, 12, 64]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_484: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_120: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_567: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_120, [384, 512, 64]);  clone_120 = None
        bmm_64: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_485, view_567);  permute_485 = None
        bmm_65: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_567, permute_486);  view_567 = permute_486 = None
        view_568: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [32, 12, 512, 64]);  bmm_64 = None
        view_569: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, [32, 12, 512, 512]);  bmm_65 = None
        convert_element_type_33: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_530: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_531: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_569, mul_530);  view_569 = mul_530 = None
        mul_532: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_531, where_3);  mul_531 = None
        sum_184: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_11: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_10: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_532);  neg_11 = sum_184 = mul_532 = None
        view_570: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_10, [384, 512, 512]);  fma_10 = None
        bmm_66: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_487, view_570);  permute_487 = None
        bmm_67: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_570, permute_488);  view_570 = permute_488 = None
        view_571: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, [32, 12, 64, 512]);  bmm_66 = None
        view_572: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [32, 12, 512, 64]);  bmm_67 = None
        mul_533: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_571, 0.3535533905932738);  view_571 = None
        permute_489: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_533, [0, 1, 3, 2]);  mul_533 = None
        mul_534: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_572, 0.3535533905932738);  view_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_490: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        clone_122: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_573: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_122, [32, 512, 768]);  clone_122 = None
        view_574: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_573, [16384, 768]);  view_573 = None
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_130: "f32[16384, 768]" = torch.ops.aten.mm.default(view_574, permute_491);  permute_491 = None
        permute_492: "f32[768, 16384]" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_492, view_22);  permute_492 = None
        sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
        view_575: "f32[768]" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
        view_576: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_130, [32, 512, 768]);  mm_130 = None
        add_171: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_526, view_576);  mul_526 = view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_495: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        view_577: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_495, [32, 512, 768]);  permute_495 = None
        clone_123: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_577, memory_format = torch.contiguous_format);  view_577 = None
        view_578: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_123, [16384, 768]);  clone_123 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_132: "f32[16384, 768]" = torch.ops.aten.mm.default(view_578, permute_496);  permute_496 = None
        permute_497: "f32[768, 16384]" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = None
        sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True);  view_578 = None
        view_579: "f32[768]" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
        view_580: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_132, [32, 512, 768]);  mm_132 = None
        add_172: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_171, view_580);  add_171 = view_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_500: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_534, [0, 2, 1, 3]);  mul_534 = None
        clone_124: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_581: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_124, [32, 512, 768]);  clone_124 = None
        view_582: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_581, [16384, 768]);  view_581 = None
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_134: "f32[16384, 768]" = torch.ops.aten.mm.default(view_582, permute_501);  permute_501 = None
        permute_502: "f32[768, 16384]" = torch.ops.aten.permute.default(view_582, [1, 0])
        mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
        sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
        view_583: "f32[768]" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None
        view_584: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_134, [32, 512, 768]);  mm_134 = None
        add_173: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_584);  add_172 = view_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_536: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_23);  primals_23 = None
        mul_537: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_536, 768)
        sum_188: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
        mul_538: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_536, mul_17);  mul_536 = None
        sum_189: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_189);  sum_189 = None
        sub_111: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_537, sum_188);  mul_537 = sum_188 = None
        sub_112: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_539);  sub_111 = mul_539 = None
        mul_540: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_112);  div_37 = sub_112 = None
        mul_541: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, mul_17);  mul_17 = None
        sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
        sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_542: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_543: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_540, mul_542);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_585: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_543, [16384, 768]);  mul_543 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_505: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_136: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_585, permute_505);  permute_505 = None
        permute_506: "f32[768, 16384]" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
        sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
        view_586: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        view_587: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_136, [32, 512, 3072]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_545: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_546: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_547: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
        exp_26: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
        mul_548: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_549: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_548);  view_19 = mul_548 = None
        add_175: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
        mul_550: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_587, add_175);  view_587 = add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_588: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_550, [16384, 3072]);  mul_550 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_509: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_138: "f32[16384, 768]" = torch.ops.aten.mm.default(view_588, permute_509);  permute_509 = None
        permute_510: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_588, [1, 0])
        mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
        sum_193: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
        view_589: "f32[3072]" = torch.ops.aten.reshape.default(sum_193, [3072]);  sum_193 = None
        view_590: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_138, [32, 512, 768]);  mm_138 = None
        add_176: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_540, view_590);  mul_540 = view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_552: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_17);  primals_17 = None
        mul_553: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_552, 768)
        sum_194: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        mul_554: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_552, mul_10);  mul_552 = None
        sum_195: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_195);  sum_195 = None
        sub_114: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_553, sum_194);  mul_553 = sum_194 = None
        sub_115: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_555);  sub_114 = mul_555 = None
        mul_556: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_115);  div_38 = sub_115 = None
        mul_557: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_10);  mul_10 = None
        sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_558: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_559: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_556, mul_558);  mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_591: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_559, [16384, 768]);  mul_559 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_140: "f32[16384, 768]" = torch.ops.aten.mm.default(view_591, permute_513);  permute_513 = None
        permute_514: "f32[768, 16384]" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
        sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
        view_592: "f32[768]" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
        view_593: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_140, [32, 512, 768]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_594: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_593, [32, 512, 12, 64]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_517: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_127: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_595: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_127, [384, 512, 64]);  clone_127 = None
        bmm_68: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(permute_518, view_595);  permute_518 = None
        bmm_69: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_595, permute_519);  view_595 = permute_519 = None
        view_596: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [32, 12, 512, 64]);  bmm_68 = None
        view_597: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, [32, 12, 512, 512]);  bmm_69 = None
        convert_element_type_36: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_560: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_561: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_597, mul_560);  view_597 = mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [32, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  expand_2 = full_default = None
        view_11: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [32, 12, 512, 512]);  bmm = None
        add_6: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        sub_1: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_2: "f32[32, 12, 512, 512]" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        mul_562: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_561, where_1);  mul_561 = None
        sum_199: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [-1], True)
        neg_12: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_11: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_12, sum_199, mul_562);  neg_12 = sum_199 = mul_562 = None
        view_598: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(fma_11, [384, 512, 512]);  fma_11 = None
        bmm_70: "f32[384, 64, 512]" = torch.ops.aten.bmm.default(permute_520, view_598);  permute_520 = None
        bmm_71: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_598, permute_521);  view_598 = permute_521 = None
        view_599: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, [32, 12, 64, 512]);  bmm_70 = None
        view_600: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [32, 12, 512, 64]);  bmm_71 = None
        mul_563: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_599, 0.3535533905932738);  view_599 = None
        permute_522: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_563, [0, 1, 3, 2]);  mul_563 = None
        mul_564: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_600, 0.3535533905932738);  view_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_523: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
        clone_129: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_601: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_129, [32, 512, 768]);  clone_129 = None
        view_602: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_601, [16384, 768]);  view_601 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_524: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_142: "f32[16384, 768]" = torch.ops.aten.mm.default(view_602, permute_524);  permute_524 = None
        permute_525: "f32[768, 16384]" = torch.ops.aten.permute.default(view_602, [1, 0])
        mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_525, view);  permute_525 = None
        sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
        view_603: "f32[768]" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
        view_604: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [32, 512, 768]);  mm_142 = None
        add_177: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_556, view_604);  mul_556 = view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_528: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        view_605: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_528, [32, 512, 768]);  permute_528 = None
        clone_130: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_605, memory_format = torch.contiguous_format);  view_605 = None
        view_606: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_130, [16384, 768]);  clone_130 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_144: "f32[16384, 768]" = torch.ops.aten.mm.default(view_606, permute_529);  permute_529 = None
        permute_530: "f32[768, 16384]" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = None
        sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
        view_607: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
        view_608: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [32, 512, 768]);  mm_144 = None
        add_178: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_177, view_608);  add_177 = view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_533: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_564, [0, 2, 1, 3]);  mul_564 = None
        clone_131: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_609: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_131, [32, 512, 768]);  clone_131 = None
        view_610: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_609, [16384, 768]);  view_609 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_146: "f32[16384, 768]" = torch.ops.aten.mm.default(view_610, permute_534);  permute_534 = None
        permute_535: "f32[768, 16384]" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
        sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_610, [0], True);  view_610 = None
        view_611: "f32[768]" = torch.ops.aten.reshape.default(sum_202, [768]);  sum_202 = None
        view_612: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [32, 512, 768]);  mm_146 = None
        add_179: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_178, view_612);  add_178 = view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:111 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_37: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_565: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_566: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_179, mul_565);  add_179 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_568: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_566, primals_7);  primals_7 = None
        mul_569: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_568, 768)
        sum_203: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True)
        mul_570: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_568, mul);  mul_568 = None
        sum_204: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True);  mul_570 = None
        mul_571: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, sum_204);  sum_204 = None
        sub_117: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_569, sum_203);  mul_569 = sum_203 = None
        sub_118: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_571);  sub_117 = mul_571 = None
        mul_572: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_118);  div_39 = sub_118 = None
        mul_573: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_566, mul);  mul = None
        sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 1]);  mul_573 = None
        sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1]);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        sum_207: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_572, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_12: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_12, -1);  eq_12 = None
        where_28: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, sum_207);  unsqueeze_5 = sum_207 = None
        full_default_42: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_42, [primals_2], where_28, True);  full_default_42 = primals_2 = where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(gather, [32, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_13: "b8[32, 512]" = torch.ops.aten.eq.Scalar(expand_1, -1)
        unsqueeze_6: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_13, -1);  eq_13 = None
        where_29: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_6, full_default_1, mul_572);  unsqueeze_6 = None
        full_default_44: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_44, [expand_1], where_29, True);  full_default_44 = expand_1 = where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_14: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_7: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_14, -1);  eq_14 = None
        where_30: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_7, full_default_1, mul_572);  unsqueeze_7 = full_default_1 = mul_572 = None
        full_default_46: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_46, [primals_1], where_30, True);  full_default_46 = primals_1 = where_30 = None
        add_180: "f32[30522, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_2);  slice_tensor = index_put_2 = None
        return (None, None, None, add_180, index_put_1, index_put, sum_205, sum_206, mm_147, view_611, mm_145, view_607, mm_143, view_603, mm_141, view_592, sum_196, sum_197, mm_139, view_589, mm_137, view_586, sum_190, sum_191, mm_135, view_583, mm_133, view_579, mm_131, view_575, mm_129, view_564, sum_181, sum_182, mm_127, view_561, mm_125, view_558, sum_175, sum_176, mm_123, view_555, mm_121, view_551, mm_119, view_547, mm_117, view_536, sum_166, sum_167, mm_115, view_533, mm_113, view_530, sum_160, sum_161, mm_111, view_527, mm_109, view_523, mm_107, view_519, mm_105, view_508, sum_151, sum_152, mm_103, view_505, mm_101, view_502, sum_145, sum_146, mm_99, view_499, mm_97, view_495, mm_95, view_491, mm_93, view_480, sum_136, sum_137, mm_91, view_477, mm_89, view_474, sum_130, sum_131, mm_87, view_471, mm_85, view_467, mm_83, view_463, mm_81, view_452, sum_121, sum_122, mm_79, view_449, mm_77, view_446, sum_115, sum_116, mm_75, view_443, mm_73, view_439, mm_71, view_435, mm_69, view_424, sum_106, sum_107, mm_67, view_421, mm_65, view_418, sum_100, sum_101, mm_63, view_415, mm_61, view_411, mm_59, view_407, mm_57, view_396, sum_91, sum_92, mm_55, view_393, mm_53, view_390, sum_85, sum_86, mm_51, view_387, mm_49, view_383, mm_47, view_379, mm_45, view_368, sum_76, sum_77, mm_43, view_365, mm_41, view_362, sum_70, sum_71, mm_39, view_359, mm_37, view_355, mm_35, view_351, mm_33, view_340, sum_61, sum_62, mm_31, view_337, mm_29, view_334, sum_55, sum_56, mm_27, view_331, mm_25, view_327, mm_23, view_323, mm_21, view_312, sum_46, sum_47, mm_19, view_309, mm_17, view_306, sum_40, sum_41, mm_15, view_303, mm_13, view_299, mm_11, view_295, mm_9, view_284, sum_31, sum_32, mm_7, view_281, mm_5, view_278, sum_25, sum_26, mm_3, view_275, sum_20, sum_21, view_272, None)
