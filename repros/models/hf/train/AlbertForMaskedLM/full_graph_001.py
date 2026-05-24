import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 512]", primals_2: "i64[1, 512]", primals_4: "f32[30000, 128]", primals_7: "f32[128]", primals_9: "f32[4096, 128]", primals_11: "f32[4096, 4096]", primals_13: "f32[4096, 4096]", primals_15: "f32[4096, 4096]", primals_17: "f32[4096, 4096]", primals_19: "f32[4096]", primals_21: "f32[16384, 4096]", primals_23: "f32[4096, 16384]", primals_25: "f32[4096]", primals_27: "f32[128, 4096]", primals_29: "f32[128]", primals_32: "i64[8, 512]", gather: "i64[1, 512]", mul: "f32[8, 512, 128]", view: "f32[4096, 128]", view_2: "f32[4096, 4096]", where_1: "f32[8, 64, 512, 512]", view_18: "f32[4096, 4096]", mul_4: "f32[8, 512, 4096]", view_20: "f32[4096, 4096]", addmm_5: "f32[4096, 16384]", view_22: "f32[4096, 16384]", mul_10: "f32[8, 512, 4096]", view_24: "f32[4096, 4096]", where_3: "f32[8, 64, 512, 512]", view_40: "f32[4096, 4096]", mul_14: "f32[8, 512, 4096]", view_42: "f32[4096, 4096]", addmm_11: "f32[4096, 16384]", view_44: "f32[4096, 16384]", mul_20: "f32[8, 512, 4096]", view_46: "f32[4096, 4096]", where_5: "f32[8, 64, 512, 512]", view_62: "f32[4096, 4096]", mul_24: "f32[8, 512, 4096]", view_64: "f32[4096, 4096]", addmm_17: "f32[4096, 16384]", view_66: "f32[4096, 16384]", mul_30: "f32[8, 512, 4096]", view_68: "f32[4096, 4096]", where_7: "f32[8, 64, 512, 512]", view_84: "f32[4096, 4096]", mul_34: "f32[8, 512, 4096]", view_86: "f32[4096, 4096]", addmm_23: "f32[4096, 16384]", view_88: "f32[4096, 16384]", mul_40: "f32[8, 512, 4096]", view_90: "f32[4096, 4096]", where_9: "f32[8, 64, 512, 512]", view_106: "f32[4096, 4096]", mul_44: "f32[8, 512, 4096]", view_108: "f32[4096, 4096]", addmm_29: "f32[4096, 16384]", view_110: "f32[4096, 16384]", mul_50: "f32[8, 512, 4096]", view_112: "f32[4096, 4096]", where_11: "f32[8, 64, 512, 512]", view_128: "f32[4096, 4096]", mul_54: "f32[8, 512, 4096]", view_130: "f32[4096, 4096]", addmm_35: "f32[4096, 16384]", view_132: "f32[4096, 16384]", mul_60: "f32[8, 512, 4096]", view_134: "f32[4096, 4096]", where_13: "f32[8, 64, 512, 512]", view_150: "f32[4096, 4096]", mul_64: "f32[8, 512, 4096]", view_152: "f32[4096, 4096]", addmm_41: "f32[4096, 16384]", view_154: "f32[4096, 16384]", mul_70: "f32[8, 512, 4096]", view_156: "f32[4096, 4096]", where_15: "f32[8, 64, 512, 512]", view_172: "f32[4096, 4096]", mul_74: "f32[8, 512, 4096]", view_174: "f32[4096, 4096]", addmm_47: "f32[4096, 16384]", view_176: "f32[4096, 16384]", mul_80: "f32[8, 512, 4096]", view_178: "f32[4096, 4096]", where_17: "f32[8, 64, 512, 512]", view_194: "f32[4096, 4096]", mul_84: "f32[8, 512, 4096]", view_196: "f32[4096, 4096]", addmm_53: "f32[4096, 16384]", view_198: "f32[4096, 16384]", mul_90: "f32[8, 512, 4096]", view_200: "f32[4096, 4096]", where_19: "f32[8, 64, 512, 512]", view_216: "f32[4096, 4096]", mul_94: "f32[8, 512, 4096]", view_218: "f32[4096, 4096]", addmm_59: "f32[4096, 16384]", view_220: "f32[4096, 16384]", mul_100: "f32[8, 512, 4096]", view_222: "f32[4096, 4096]", where_21: "f32[8, 64, 512, 512]", view_238: "f32[4096, 4096]", mul_104: "f32[8, 512, 4096]", view_240: "f32[4096, 4096]", addmm_65: "f32[4096, 16384]", view_242: "f32[4096, 16384]", mul_110: "f32[8, 512, 4096]", view_244: "f32[4096, 4096]", where_23: "f32[8, 64, 512, 512]", view_260: "f32[4096, 4096]", mul_114: "f32[8, 512, 4096]", view_262: "f32[4096, 4096]", addmm_71: "f32[4096, 16384]", view_264: "f32[4096, 16384]", mul_120: "f32[8, 512, 4096]", view_266: "f32[4096, 4096]", addmm_73: "f32[4096, 128]", getitem_51: "f32[8, 512, 1]", rsqrt_25: "f32[8, 512, 1]", view_268: "f32[4096, 128]", view_269: "f32[8, 512, 30000]", amax_12: "f32[4096, 1]", log: "f32[4096, 1]", convert_element_type: "f32[]", div_15: "f32[8, 512, 1]", div_16: "f32[8, 512, 1]", permute_157: "f32[512, 64, 512]", permute_158: "f32[512, 64, 512]", permute_159: "f32[512, 512, 64]", div_17: "f32[8, 512, 1]", div_18: "f32[8, 512, 1]", permute_190: "f32[512, 64, 512]", permute_191: "f32[512, 64, 512]", permute_192: "f32[512, 512, 64]", div_19: "f32[8, 512, 1]", div_20: "f32[8, 512, 1]", permute_223: "f32[512, 64, 512]", permute_224: "f32[512, 64, 512]", permute_225: "f32[512, 512, 64]", div_21: "f32[8, 512, 1]", div_22: "f32[8, 512, 1]", permute_256: "f32[512, 64, 512]", permute_257: "f32[512, 64, 512]", permute_258: "f32[512, 512, 64]", div_23: "f32[8, 512, 1]", div_24: "f32[8, 512, 1]", permute_289: "f32[512, 64, 512]", permute_290: "f32[512, 64, 512]", permute_291: "f32[512, 512, 64]", div_25: "f32[8, 512, 1]", div_26: "f32[8, 512, 1]", permute_322: "f32[512, 64, 512]", permute_323: "f32[512, 64, 512]", permute_324: "f32[512, 512, 64]", div_27: "f32[8, 512, 1]", div_28: "f32[8, 512, 1]", permute_355: "f32[512, 64, 512]", permute_356: "f32[512, 64, 512]", permute_357: "f32[512, 512, 64]", div_29: "f32[8, 512, 1]", div_30: "f32[8, 512, 1]", permute_388: "f32[512, 64, 512]", permute_389: "f32[512, 64, 512]", permute_390: "f32[512, 512, 64]", div_31: "f32[8, 512, 1]", div_32: "f32[8, 512, 1]", permute_421: "f32[512, 64, 512]", permute_422: "f32[512, 64, 512]", permute_423: "f32[512, 512, 64]", div_33: "f32[8, 512, 1]", div_34: "f32[8, 512, 1]", permute_454: "f32[512, 64, 512]", permute_455: "f32[512, 64, 512]", permute_456: "f32[512, 512, 64]", div_35: "f32[8, 512, 1]", div_36: "f32[8, 512, 1]", permute_487: "f32[512, 64, 512]", permute_488: "f32[512, 64, 512]", permute_489: "f32[512, 512, 64]", div_37: "f32[8, 512, 1]", div_38: "f32[8, 512, 1]", permute_520: "f32[512, 64, 512]", permute_521: "f32[512, 64, 512]", permute_522: "f32[512, 512, 64]", div_39: "f32[8, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[8, 512, 30000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:650 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_271: "i64[4096]" = torch.ops.aten.reshape.default(primals_32, [-1]);  primals_32 = None
        unsqueeze_4: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view_271, 1);  view_271 = None
        ne_3: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[4096, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_36);  unsqueeze_4 = full_default_36 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30000]" = torch.ops.prims.iota.default(30000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30000]" = torch.ops.aten.reshape.default(iota_default, [1, 30000]);  iota_default = None
        expand_default: "i64[4096, 30000]" = torch.ops.aten.expand.default(where_26, [4096, 30000]);  where_26 = None
        eq_tensor: "b8[4096, 30000]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:650 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[4096, 30000]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:650 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_27: "f32[4096, 1]" = torch.ops.aten.where.self(ne_3, div_13, full_default_1);  ne_3 = div_13 = None
        mul_128: "f32[4096, 30000]" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None
        view_270: "f32[4096, 30000]" = torch.ops.aten.reshape.default(view_269, [-1, 30000]);  view_269 = None
        sub_38: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = amax_12 = None
        sub_39: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        exp_13: "f32[4096, 30000]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_16: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [1], True)
        mul_129: "f32[4096, 30000]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_40: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
        view_272: "f32[8, 512, 30000]" = torch.ops.aten.reshape.default(sub_40, [8, 512, 30000]);  sub_40 = None
        add_118: "f32[8, 512, 30000]" = torch.ops.aten.add.Tensor(tangents_2, view_272);  tangents_2 = view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        view_273: "f32[4096, 30000]" = torch.ops.aten.reshape.default(add_118, [4096, 30000]);  add_118 = None
        permute_134: "f32[128, 30000]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_135: "f32[30000, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm: "f32[4096, 128]" = torch.ops.aten.mm.default(view_273, permute_135);  permute_135 = None
        permute_136: "f32[30000, 4096]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_1: "f32[30000, 128]" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
        sum_17: "f32[1, 30000]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        view_274: "f32[30000]" = torch.ops.aten.reshape.default(sum_17, [30000]);  sum_17 = None
        view_275: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(mm, [8, 512, 128]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_131: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, primals_29);  primals_29 = None
        mul_132: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_131, 128)
        sum_18: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        view_267: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [8, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_122: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        pow_13: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 3.0)
        mul_123: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_114: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view_267, mul_123);  mul_123 = None
        mul_124: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_114, 0.7978845608028654);  add_114 = None
        tanh_12: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_124);  mul_124 = None
        add_115: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_125: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_122, add_115)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_37: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_125, getitem_51);  mul_125 = getitem_51 = None
        mul_126: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_133: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_131, mul_126);  mul_131 = None
        sum_19: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True);  mul_133 = None
        mul_134: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_126, sum_19);  sum_19 = None
        sub_42: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_132, sum_18);  mul_132 = sum_18 = None
        sub_43: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_42, mul_134);  sub_42 = mul_134 = None
        div_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_135: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_14, sub_43);  div_14 = sub_43 = None
        mul_136: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_126);  mul_126 = None
        sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_136, [0, 1]);  mul_136 = None
        sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_137: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_135, mul_122);  mul_122 = None
        mul_138: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_135, add_115);  mul_135 = add_115 = None
        mul_139: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_44: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_139);  mul_139 = None
        mul_140: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_137, sub_44);  mul_137 = sub_44 = None
        mul_141: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_140, 0.7978845608028654);  mul_140 = None
        mul_142: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_141, 0.044715)
        pow_14: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 2.0);  view_267 = None
        mul_143: "f32[8, 512, 128]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_144: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        add_119: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_141, mul_144);  mul_141 = mul_144 = None
        mul_145: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_138, 0.5);  mul_138 = None
        add_120: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(add_119, mul_145);  add_119 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        view_276: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_120, [4096, 128]);  add_120 = None
        permute_133: "f32[4096, 128]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_139: "f32[128, 4096]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_2: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_276, permute_139);  permute_139 = None
        permute_140: "f32[128, 4096]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_3: "f32[128, 4096]" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
        sum_22: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        view_277: "f32[128]" = torch.ops.aten.reshape.default(sum_22, [128]);  sum_22 = None
        view_278: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_2, [8, 512, 4096]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_147: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_278, primals_25)
        mul_148: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_147, 4096)
        sum_23: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True)
        mul_149: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_147, mul_120);  mul_147 = None
        sum_24: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
        mul_150: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_120, sum_24);  sum_24 = None
        sub_46: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_148, sum_23);  mul_148 = sum_23 = None
        sub_47: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_46, mul_150);  sub_46 = mul_150 = None
        mul_151: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_15, sub_47);  div_15 = sub_47 = None
        mul_152: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_278, mul_120);  mul_120 = None
        sum_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_152, [0, 1]);  mul_152 = None
        sum_26: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_279: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_151, [4096, 4096])
        permute_11: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_143: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_4: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_279, permute_143)
        permute_144: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_5: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
        sum_27: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        view_280: "f32[4096]" = torch.ops.aten.reshape.default(sum_27, [4096]);  sum_27 = None
        view_281: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_4, [8, 512, 16384]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_263: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_71, [8, 512, 16384]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_153: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_281, mul_116);  mul_116 = None
        pow_12: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
        mul_117: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_263, mul_117);  mul_117 = None
        mul_118: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_110: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_154: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_281, add_110);  view_281 = add_110 = None
        mul_155: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_48: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_155);  mul_155 = None
        mul_156: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_153, sub_48);  mul_153 = sub_48 = None
        mul_157: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_156, 0.7978845608028654);  mul_156 = None
        mul_158: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_157, 0.044715)
        pow_15: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 2.0);  view_263 = None
        mul_159: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_160: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        add_121: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_157, mul_160);  mul_157 = mul_160 = None
        mul_161: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_154, 0.5);  mul_154 = None
        add_122: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_121, mul_161);  add_121 = mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_282: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_122, [4096, 16384]);  add_122 = None
        permute_10: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_147: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_6: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_282, permute_147)
        permute_148: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_7: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
        sum_28: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283: "f32[16384]" = torch.ops.aten.reshape.default(sum_28, [16384]);  sum_28 = None
        view_284: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_6, [8, 512, 4096]);  mm_6 = None
        add_123: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_151, view_284);  mul_151 = view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_163: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_123, primals_19)
        mul_164: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_163, 4096)
        sum_29: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
        mul_165: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_163, mul_114);  mul_163 = None
        sum_30: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
        mul_166: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_114, sum_30);  sum_30 = None
        sub_50: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_164, sum_29);  mul_164 = sum_29 = None
        sub_51: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_50, mul_166);  sub_50 = mul_166 = None
        mul_167: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_16, sub_51);  div_16 = sub_51 = None
        mul_168: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_123, mul_114);  mul_114 = None
        sum_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
        sum_32: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_285: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_167, [4096, 4096])
        permute_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_151: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_8: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_285, permute_151)
        permute_152: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_9: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
        sum_33: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_286: "f32[4096]" = torch.ops.aten.reshape.default(sum_33, [4096]);  sum_33 = None
        view_287: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_8, [8, 512, 4096]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_288: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_287, [8, 512, 64, 64]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_61: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_289: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_61, [512, 512, 64]);  clone_61 = None
        expand_49: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_23, [8, 64, 512, 512])
        view_256: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [512, 512, 512]);  expand_49 = None
        permute_156: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        bmm_24: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_156, view_289);  permute_156 = None
        bmm_25: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_289, permute_157);  view_289 = permute_157 = None
        view_290: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_24, [8, 64, 512, 64]);  bmm_24 = None
        view_291: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_25, [8, 64, 512, 512]);  bmm_25 = None
        mul_169: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_291, where_23);  view_291 = None
        sum_34: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [-1], True)
        neg_1: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_23);  where_23 = None
        fma: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_169);  neg_1 = sum_34 = mul_169 = None
        view_292: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma, [512, 512, 512]);  fma = None
        bmm_26: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_158, view_292);  permute_158 = None
        bmm_27: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_292, permute_159);  view_292 = permute_159 = None
        view_293: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_26, [8, 64, 64, 512]);  bmm_26 = None
        view_294: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [8, 64, 512, 64]);  bmm_27 = None
        mul_170: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_293, 0.3535533905932738);  view_293 = None
        permute_160: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_170, [0, 1, 3, 2]);  mul_170 = None
        mul_171: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_294, 0.3535533905932738);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_161: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_62: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_295: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_62, [8, 512, 4096]);  clone_62 = None
        view_296: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_295, [4096, 4096]);  view_295 = None
        permute_5: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_162: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_10: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_296, permute_162)
        permute_163: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_11: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_163, view_244);  permute_163 = None
        sum_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        view_297: "f32[4096]" = torch.ops.aten.reshape.default(sum_35, [4096]);  sum_35 = None
        view_298: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_10, [8, 512, 4096]);  mm_10 = None
        add_124: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_167, view_298);  mul_167 = view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_166: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
        view_299: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_166, [8, 512, 4096]);  permute_166 = None
        clone_63: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_299, memory_format = torch.contiguous_format);  view_299 = None
        view_300: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_63, [4096, 4096]);  clone_63 = None
        permute_3: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_167: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_12: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_300, permute_167)
        permute_168: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_13: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = None
        sum_36: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        view_301: "f32[4096]" = torch.ops.aten.reshape.default(sum_36, [4096]);  sum_36 = None
        view_302: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_12, [8, 512, 4096]);  mm_12 = None
        add_125: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_124, view_302);  add_124 = view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_171: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_171, [0, 2, 1, 3]);  mul_171 = None
        clone_64: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_303: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_64, [8, 512, 4096]);  clone_64 = None
        view_304: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_303, [4096, 4096]);  view_303 = None
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_172: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_14: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_304, permute_172)
        permute_173: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_15: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
        sum_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        view_305: "f32[4096]" = torch.ops.aten.reshape.default(sum_37, [4096]);  sum_37 = None
        view_306: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_14, [8, 512, 4096]);  mm_14 = None
        add_126: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_125, view_306);  add_125 = view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_173: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_126, primals_25)
        mul_174: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_173, 4096)
        sum_38: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True)
        mul_175: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_173, mul_110);  mul_173 = None
        sum_39: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
        mul_176: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_110, sum_39);  sum_39 = None
        sub_53: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_174, sum_38);  mul_174 = sum_38 = None
        sub_54: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_53, mul_176);  sub_53 = mul_176 = None
        mul_177: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_17, sub_54);  div_17 = sub_54 = None
        mul_178: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_126, mul_110);  mul_110 = None
        sum_40: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
        sum_41: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
        add_127: "f32[4096]" = torch.ops.aten.add.Tensor(sum_25, sum_40);  sum_25 = sum_40 = None
        add_128: "f32[4096]" = torch.ops.aten.add.Tensor(sum_26, sum_41);  sum_26 = sum_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_307: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_177, [4096, 4096])
        mm_16: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_307, permute_143)
        permute_177: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_307, [1, 0])
        mm_17: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
        sum_42: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        view_308: "f32[4096]" = torch.ops.aten.reshape.default(sum_42, [4096]);  sum_42 = None
        add_129: "f32[4096]" = torch.ops.aten.add.Tensor(view_280, view_308);  view_280 = view_308 = None
        add_130: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(mm_5, mm_17);  mm_5 = mm_17 = None
        view_309: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_16, [8, 512, 16384]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_241: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_65, [8, 512, 16384]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_179: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_309, mul_106);  mul_106 = None
        pow_11: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
        mul_107: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_241, mul_107);  mul_107 = None
        mul_108: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_101: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_180: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_309, add_101);  view_309 = add_101 = None
        mul_181: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_55: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_181);  mul_181 = None
        mul_182: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_179, sub_55);  mul_179 = sub_55 = None
        mul_183: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_182, 0.7978845608028654);  mul_182 = None
        mul_184: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_183, 0.044715)
        pow_16: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 2.0);  view_241 = None
        mul_185: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_186: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
        add_131: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_183, mul_186);  mul_183 = mul_186 = None
        mul_187: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_180, 0.5);  mul_180 = None
        add_132: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_131, mul_187);  add_131 = mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_310: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_132, [4096, 16384]);  add_132 = None
        mm_18: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_310, permute_147)
        permute_181: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_19: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
        sum_43: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        view_311: "f32[16384]" = torch.ops.aten.reshape.default(sum_43, [16384]);  sum_43 = None
        add_133: "f32[16384]" = torch.ops.aten.add.Tensor(view_283, view_311);  view_283 = view_311 = None
        add_134: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(mm_7, mm_19);  mm_7 = mm_19 = None
        view_312: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_18, [8, 512, 4096]);  mm_18 = None
        add_135: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_177, view_312);  mul_177 = view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_189: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_135, primals_19)
        mul_190: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_189, 4096)
        sum_44: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True)
        mul_191: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_189, mul_104);  mul_189 = None
        sum_45: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True);  mul_191 = None
        mul_192: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_104, sum_45);  sum_45 = None
        sub_57: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_190, sum_44);  mul_190 = sum_44 = None
        sub_58: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_57, mul_192);  sub_57 = mul_192 = None
        mul_193: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_18, sub_58);  div_18 = sub_58 = None
        mul_194: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_135, mul_104);  mul_104 = None
        sum_46: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 1]);  mul_194 = None
        sum_47: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
        add_136: "f32[4096]" = torch.ops.aten.add.Tensor(sum_31, sum_46);  sum_31 = sum_46 = None
        add_137: "f32[4096]" = torch.ops.aten.add.Tensor(sum_32, sum_47);  sum_32 = sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_313: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_193, [4096, 4096])
        mm_20: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_313, permute_151)
        permute_185: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_21: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
        sum_48: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
        view_314: "f32[4096]" = torch.ops.aten.reshape.default(sum_48, [4096]);  sum_48 = None
        add_138: "f32[4096]" = torch.ops.aten.add.Tensor(view_286, view_314);  view_286 = view_314 = None
        add_139: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_9, mm_21);  mm_9 = mm_21 = None
        view_315: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_20, [8, 512, 4096]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_316: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_315, [8, 512, 64, 64]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_188: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_65: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_317: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_65, [512, 512, 64]);  clone_65 = None
        expand_45: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_21, [8, 64, 512, 512])
        view_234: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [512, 512, 512]);  expand_45 = None
        permute_189: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        bmm_28: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_189, view_317);  permute_189 = None
        bmm_29: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_317, permute_190);  view_317 = permute_190 = None
        view_318: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_28, [8, 64, 512, 64]);  bmm_28 = None
        view_319: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_29, [8, 64, 512, 512]);  bmm_29 = None
        mul_195: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_319, where_21);  view_319 = None
        sum_49: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [-1], True)
        neg_2: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_21);  where_21 = None
        fma_1: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_195);  neg_2 = sum_49 = mul_195 = None
        view_320: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_1, [512, 512, 512]);  fma_1 = None
        bmm_30: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_191, view_320);  permute_191 = None
        bmm_31: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_320, permute_192);  view_320 = permute_192 = None
        view_321: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_30, [8, 64, 64, 512]);  bmm_30 = None
        view_322: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [8, 64, 512, 64]);  bmm_31 = None
        mul_196: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_321, 0.3535533905932738);  view_321 = None
        permute_193: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_196, [0, 1, 3, 2]);  mul_196 = None
        mul_197: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_322, 0.3535533905932738);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_194: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_66: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_323: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_66, [8, 512, 4096]);  clone_66 = None
        view_324: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_323, [4096, 4096]);  view_323 = None
        mm_22: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_324, permute_162)
        permute_196: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_23: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_196, view_222);  permute_196 = None
        sum_50: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325: "f32[4096]" = torch.ops.aten.reshape.default(sum_50, [4096]);  sum_50 = None
        add_140: "f32[4096]" = torch.ops.aten.add.Tensor(view_297, view_325);  view_297 = view_325 = None
        add_141: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_11, mm_23);  mm_11 = mm_23 = None
        view_326: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_22, [8, 512, 4096]);  mm_22 = None
        add_142: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_193, view_326);  mul_193 = view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_199: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
        view_327: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_199, [8, 512, 4096]);  permute_199 = None
        clone_67: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_327, memory_format = torch.contiguous_format);  view_327 = None
        view_328: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_67, [4096, 4096]);  clone_67 = None
        mm_24: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_328, permute_167)
        permute_201: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_25: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = None
        sum_51: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        view_329: "f32[4096]" = torch.ops.aten.reshape.default(sum_51, [4096]);  sum_51 = None
        add_143: "f32[4096]" = torch.ops.aten.add.Tensor(view_301, view_329);  view_301 = view_329 = None
        add_144: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_13, mm_25);  mm_13 = mm_25 = None
        view_330: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_24, [8, 512, 4096]);  mm_24 = None
        add_145: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_142, view_330);  add_142 = view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_204: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_197, [0, 2, 1, 3]);  mul_197 = None
        clone_68: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_331: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_68, [8, 512, 4096]);  clone_68 = None
        view_332: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_331, [4096, 4096]);  view_331 = None
        mm_26: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_332, permute_172)
        permute_206: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_27: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
        sum_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
        view_333: "f32[4096]" = torch.ops.aten.reshape.default(sum_52, [4096]);  sum_52 = None
        add_146: "f32[4096]" = torch.ops.aten.add.Tensor(view_305, view_333);  view_305 = view_333 = None
        add_147: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_15, mm_27);  mm_15 = mm_27 = None
        view_334: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_26, [8, 512, 4096]);  mm_26 = None
        add_148: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_145, view_334);  add_145 = view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_199: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_148, primals_25)
        mul_200: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_199, 4096)
        sum_53: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True)
        mul_201: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_199, mul_100);  mul_199 = None
        sum_54: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True);  mul_201 = None
        mul_202: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_100, sum_54);  sum_54 = None
        sub_60: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_200, sum_53);  mul_200 = sum_53 = None
        sub_61: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_60, mul_202);  sub_60 = mul_202 = None
        mul_203: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_19, sub_61);  div_19 = sub_61 = None
        mul_204: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_148, mul_100);  mul_100 = None
        sum_55: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1]);  mul_204 = None
        sum_56: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None
        add_149: "f32[4096]" = torch.ops.aten.add.Tensor(add_127, sum_55);  add_127 = sum_55 = None
        add_150: "f32[4096]" = torch.ops.aten.add.Tensor(add_128, sum_56);  add_128 = sum_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_335: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_203, [4096, 4096])
        mm_28: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_335, permute_143)
        permute_210: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_335, [1, 0])
        mm_29: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
        sum_57: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        view_336: "f32[4096]" = torch.ops.aten.reshape.default(sum_57, [4096]);  sum_57 = None
        add_151: "f32[4096]" = torch.ops.aten.add.Tensor(add_129, view_336);  add_129 = view_336 = None
        add_152: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_130, mm_29);  add_130 = mm_29 = None
        view_337: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_28, [8, 512, 16384]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_219: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_59, [8, 512, 16384]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_205: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_337, mul_96);  mul_96 = None
        pow_10: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
        mul_97: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_219, mul_97);  mul_97 = None
        mul_98: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_92: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_206: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_337, add_92);  view_337 = add_92 = None
        mul_207: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_62: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_207);  mul_207 = None
        mul_208: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_205, sub_62);  mul_205 = sub_62 = None
        mul_209: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_208, 0.7978845608028654);  mul_208 = None
        mul_210: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_209, 0.044715)
        pow_17: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 2.0);  view_219 = None
        mul_211: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_212: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
        add_153: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_209, mul_212);  mul_209 = mul_212 = None
        mul_213: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_206, 0.5);  mul_206 = None
        add_154: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_153, mul_213);  add_153 = mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_338: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_154, [4096, 16384]);  add_154 = None
        mm_30: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_338, permute_147)
        permute_214: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_31: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
        sum_58: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        view_339: "f32[16384]" = torch.ops.aten.reshape.default(sum_58, [16384]);  sum_58 = None
        add_155: "f32[16384]" = torch.ops.aten.add.Tensor(add_133, view_339);  add_133 = view_339 = None
        add_156: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_134, mm_31);  add_134 = mm_31 = None
        view_340: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_30, [8, 512, 4096]);  mm_30 = None
        add_157: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_203, view_340);  mul_203 = view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_215: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_157, primals_19)
        mul_216: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_215, 4096)
        sum_59: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
        mul_217: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_215, mul_94);  mul_215 = None
        sum_60: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
        mul_218: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_94, sum_60);  sum_60 = None
        sub_64: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_216, sum_59);  mul_216 = sum_59 = None
        sub_65: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_64, mul_218);  sub_64 = mul_218 = None
        mul_219: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_20, sub_65);  div_20 = sub_65 = None
        mul_220: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_157, mul_94);  mul_94 = None
        sum_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
        sum_62: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_157, [0, 1]);  add_157 = None
        add_158: "f32[4096]" = torch.ops.aten.add.Tensor(add_136, sum_61);  add_136 = sum_61 = None
        add_159: "f32[4096]" = torch.ops.aten.add.Tensor(add_137, sum_62);  add_137 = sum_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_341: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_219, [4096, 4096])
        mm_32: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_341, permute_151)
        permute_218: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_341, [1, 0])
        mm_33: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
        sum_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        view_342: "f32[4096]" = torch.ops.aten.reshape.default(sum_63, [4096]);  sum_63 = None
        add_160: "f32[4096]" = torch.ops.aten.add.Tensor(add_138, view_342);  add_138 = view_342 = None
        add_161: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_139, mm_33);  add_139 = mm_33 = None
        view_343: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_32, [8, 512, 4096]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_344: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_343, [8, 512, 64, 64]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_69: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_345: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_69, [512, 512, 64]);  clone_69 = None
        expand_41: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_19, [8, 64, 512, 512])
        view_212: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [512, 512, 512]);  expand_41 = None
        permute_222: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        bmm_32: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_222, view_345);  permute_222 = None
        bmm_33: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_345, permute_223);  view_345 = permute_223 = None
        view_346: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_32, [8, 64, 512, 64]);  bmm_32 = None
        view_347: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_33, [8, 64, 512, 512]);  bmm_33 = None
        mul_221: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_347, where_19);  view_347 = None
        sum_64: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [-1], True)
        neg_3: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_19);  where_19 = None
        fma_2: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_221);  neg_3 = sum_64 = mul_221 = None
        view_348: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_2, [512, 512, 512]);  fma_2 = None
        bmm_34: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_224, view_348);  permute_224 = None
        bmm_35: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_348, permute_225);  view_348 = permute_225 = None
        view_349: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_34, [8, 64, 64, 512]);  bmm_34 = None
        view_350: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [8, 64, 512, 64]);  bmm_35 = None
        mul_222: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_349, 0.3535533905932738);  view_349 = None
        permute_226: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_222, [0, 1, 3, 2]);  mul_222 = None
        mul_223: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_350, 0.3535533905932738);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_227: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_70: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_351: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_70, [8, 512, 4096]);  clone_70 = None
        view_352: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_351, [4096, 4096]);  view_351 = None
        mm_34: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_352, permute_162)
        permute_229: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_35: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_229, view_200);  permute_229 = None
        sum_65: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        view_353: "f32[4096]" = torch.ops.aten.reshape.default(sum_65, [4096]);  sum_65 = None
        add_162: "f32[4096]" = torch.ops.aten.add.Tensor(add_140, view_353);  add_140 = view_353 = None
        add_163: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_141, mm_35);  add_141 = mm_35 = None
        view_354: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_34, [8, 512, 4096]);  mm_34 = None
        add_164: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_219, view_354);  mul_219 = view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_232: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
        view_355: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_232, [8, 512, 4096]);  permute_232 = None
        clone_71: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_355, memory_format = torch.contiguous_format);  view_355 = None
        view_356: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_71, [4096, 4096]);  clone_71 = None
        mm_36: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_356, permute_167)
        permute_234: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_37: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = None
        sum_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        view_357: "f32[4096]" = torch.ops.aten.reshape.default(sum_66, [4096]);  sum_66 = None
        add_165: "f32[4096]" = torch.ops.aten.add.Tensor(add_143, view_357);  add_143 = view_357 = None
        add_166: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_144, mm_37);  add_144 = mm_37 = None
        view_358: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_36, [8, 512, 4096]);  mm_36 = None
        add_167: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_164, view_358);  add_164 = view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_237: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_223, [0, 2, 1, 3]);  mul_223 = None
        clone_72: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_359: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_72, [8, 512, 4096]);  clone_72 = None
        view_360: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_359, [4096, 4096]);  view_359 = None
        mm_38: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_360, permute_172)
        permute_239: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_39: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
        sum_67: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        view_361: "f32[4096]" = torch.ops.aten.reshape.default(sum_67, [4096]);  sum_67 = None
        add_168: "f32[4096]" = torch.ops.aten.add.Tensor(add_146, view_361);  add_146 = view_361 = None
        add_169: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_147, mm_39);  add_147 = mm_39 = None
        view_362: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_38, [8, 512, 4096]);  mm_38 = None
        add_170: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_167, view_362);  add_167 = view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_225: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_170, primals_25)
        mul_226: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_225, 4096)
        sum_68: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
        mul_227: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_225, mul_90);  mul_225 = None
        sum_69: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
        mul_228: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_90, sum_69);  sum_69 = None
        sub_67: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_226, sum_68);  mul_226 = sum_68 = None
        sub_68: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_67, mul_228);  sub_67 = mul_228 = None
        mul_229: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_21, sub_68);  div_21 = sub_68 = None
        mul_230: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_170, mul_90);  mul_90 = None
        sum_70: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
        sum_71: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
        add_171: "f32[4096]" = torch.ops.aten.add.Tensor(add_149, sum_70);  add_149 = sum_70 = None
        add_172: "f32[4096]" = torch.ops.aten.add.Tensor(add_150, sum_71);  add_150 = sum_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_363: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_229, [4096, 4096])
        mm_40: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_363, permute_143)
        permute_243: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_41: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
        sum_72: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        view_364: "f32[4096]" = torch.ops.aten.reshape.default(sum_72, [4096]);  sum_72 = None
        add_173: "f32[4096]" = torch.ops.aten.add.Tensor(add_151, view_364);  add_151 = view_364 = None
        add_174: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_152, mm_41);  add_152 = mm_41 = None
        view_365: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_40, [8, 512, 16384]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_197: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_53, [8, 512, 16384]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_231: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_365, mul_86);  mul_86 = None
        pow_9: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
        mul_87: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_197, mul_87);  mul_87 = None
        mul_88: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_83: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_232: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_365, add_83);  view_365 = add_83 = None
        mul_233: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_69: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_233);  mul_233 = None
        mul_234: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_231, sub_69);  mul_231 = sub_69 = None
        mul_235: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_234, 0.7978845608028654);  mul_234 = None
        mul_236: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_235, 0.044715)
        pow_18: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
        mul_237: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_238: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        add_175: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_235, mul_238);  mul_235 = mul_238 = None
        mul_239: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_232, 0.5);  mul_232 = None
        add_176: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_175, mul_239);  add_175 = mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_366: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_176, [4096, 16384]);  add_176 = None
        mm_42: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_366, permute_147)
        permute_247: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_43: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
        sum_73: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        view_367: "f32[16384]" = torch.ops.aten.reshape.default(sum_73, [16384]);  sum_73 = None
        add_177: "f32[16384]" = torch.ops.aten.add.Tensor(add_155, view_367);  add_155 = view_367 = None
        add_178: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_156, mm_43);  add_156 = mm_43 = None
        view_368: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_42, [8, 512, 4096]);  mm_42 = None
        add_179: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_229, view_368);  mul_229 = view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_241: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_179, primals_19)
        mul_242: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_241, 4096)
        sum_74: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
        mul_243: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_241, mul_84);  mul_241 = None
        sum_75: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
        mul_244: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_84, sum_75);  sum_75 = None
        sub_71: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_242, sum_74);  mul_242 = sum_74 = None
        sub_72: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_71, mul_244);  sub_71 = mul_244 = None
        mul_245: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_22, sub_72);  div_22 = sub_72 = None
        mul_246: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_179, mul_84);  mul_84 = None
        sum_76: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
        sum_77: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
        add_180: "f32[4096]" = torch.ops.aten.add.Tensor(add_158, sum_76);  add_158 = sum_76 = None
        add_181: "f32[4096]" = torch.ops.aten.add.Tensor(add_159, sum_77);  add_159 = sum_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_369: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_245, [4096, 4096])
        mm_44: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_369, permute_151)
        permute_251: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_45: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
        sum_78: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        view_370: "f32[4096]" = torch.ops.aten.reshape.default(sum_78, [4096]);  sum_78 = None
        add_182: "f32[4096]" = torch.ops.aten.add.Tensor(add_160, view_370);  add_160 = view_370 = None
        add_183: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_161, mm_45);  add_161 = mm_45 = None
        view_371: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_44, [8, 512, 4096]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_372: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_371, [8, 512, 64, 64]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_254: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_73: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_373: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_73, [512, 512, 64]);  clone_73 = None
        expand_37: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_17, [8, 64, 512, 512])
        view_190: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [512, 512, 512]);  expand_37 = None
        permute_255: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        bmm_36: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_255, view_373);  permute_255 = None
        bmm_37: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_373, permute_256);  view_373 = permute_256 = None
        view_374: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_36, [8, 64, 512, 64]);  bmm_36 = None
        view_375: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_37, [8, 64, 512, 512]);  bmm_37 = None
        mul_247: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_375, where_17);  view_375 = None
        sum_79: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [-1], True)
        neg_4: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_17);  where_17 = None
        fma_3: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_247);  neg_4 = sum_79 = mul_247 = None
        view_376: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_3, [512, 512, 512]);  fma_3 = None
        bmm_38: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_257, view_376);  permute_257 = None
        bmm_39: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_376, permute_258);  view_376 = permute_258 = None
        view_377: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_38, [8, 64, 64, 512]);  bmm_38 = None
        view_378: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [8, 64, 512, 64]);  bmm_39 = None
        mul_248: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_377, 0.3535533905932738);  view_377 = None
        permute_259: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_248, [0, 1, 3, 2]);  mul_248 = None
        mul_249: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_378, 0.3535533905932738);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_260: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_74: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_379: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_74, [8, 512, 4096]);  clone_74 = None
        view_380: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_379, [4096, 4096]);  view_379 = None
        mm_46: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_380, permute_162)
        permute_262: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_47: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_262, view_178);  permute_262 = None
        sum_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        view_381: "f32[4096]" = torch.ops.aten.reshape.default(sum_80, [4096]);  sum_80 = None
        add_184: "f32[4096]" = torch.ops.aten.add.Tensor(add_162, view_381);  add_162 = view_381 = None
        add_185: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_163, mm_47);  add_163 = mm_47 = None
        view_382: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_46, [8, 512, 4096]);  mm_46 = None
        add_186: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_245, view_382);  mul_245 = view_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_265: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
        view_383: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_265, [8, 512, 4096]);  permute_265 = None
        clone_75: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_383, memory_format = torch.contiguous_format);  view_383 = None
        view_384: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_75, [4096, 4096]);  clone_75 = None
        mm_48: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_384, permute_167)
        permute_267: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_384, [1, 0])
        mm_49: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = None
        sum_81: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
        view_385: "f32[4096]" = torch.ops.aten.reshape.default(sum_81, [4096]);  sum_81 = None
        add_187: "f32[4096]" = torch.ops.aten.add.Tensor(add_165, view_385);  add_165 = view_385 = None
        add_188: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_166, mm_49);  add_166 = mm_49 = None
        view_386: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_48, [8, 512, 4096]);  mm_48 = None
        add_189: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_186, view_386);  add_186 = view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_270: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_249, [0, 2, 1, 3]);  mul_249 = None
        clone_76: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_387: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_76, [8, 512, 4096]);  clone_76 = None
        view_388: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_387, [4096, 4096]);  view_387 = None
        mm_50: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_388, permute_172)
        permute_272: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_51: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
        sum_82: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        view_389: "f32[4096]" = torch.ops.aten.reshape.default(sum_82, [4096]);  sum_82 = None
        add_190: "f32[4096]" = torch.ops.aten.add.Tensor(add_168, view_389);  add_168 = view_389 = None
        add_191: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_169, mm_51);  add_169 = mm_51 = None
        view_390: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_50, [8, 512, 4096]);  mm_50 = None
        add_192: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_189, view_390);  add_189 = view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_251: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, primals_25)
        mul_252: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_251, 4096)
        sum_83: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
        mul_253: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_251, mul_80);  mul_251 = None
        sum_84: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        mul_254: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_80, sum_84);  sum_84 = None
        sub_74: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_252, sum_83);  mul_252 = sum_83 = None
        sub_75: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_74, mul_254);  sub_74 = mul_254 = None
        mul_255: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_23, sub_75);  div_23 = sub_75 = None
        mul_256: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, mul_80);  mul_80 = None
        sum_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
        sum_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1]);  add_192 = None
        add_193: "f32[4096]" = torch.ops.aten.add.Tensor(add_171, sum_85);  add_171 = sum_85 = None
        add_194: "f32[4096]" = torch.ops.aten.add.Tensor(add_172, sum_86);  add_172 = sum_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_391: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_255, [4096, 4096])
        mm_52: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_391, permute_143)
        permute_276: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_53: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
        sum_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        view_392: "f32[4096]" = torch.ops.aten.reshape.default(sum_87, [4096]);  sum_87 = None
        add_195: "f32[4096]" = torch.ops.aten.add.Tensor(add_173, view_392);  add_173 = view_392 = None
        add_196: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_174, mm_53);  add_174 = mm_53 = None
        view_393: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_52, [8, 512, 16384]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_175: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_47, [8, 512, 16384]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_257: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_393, mul_76);  mul_76 = None
        pow_8: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
        mul_77: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_175, mul_77);  mul_77 = None
        mul_78: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_74: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_258: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_393, add_74);  view_393 = add_74 = None
        mul_259: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_76: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_259);  mul_259 = None
        mul_260: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_257, sub_76);  mul_257 = sub_76 = None
        mul_261: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_260, 0.7978845608028654);  mul_260 = None
        mul_262: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_261, 0.044715)
        pow_19: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 2.0);  view_175 = None
        mul_263: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_264: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        add_197: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_261, mul_264);  mul_261 = mul_264 = None
        mul_265: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_258, 0.5);  mul_258 = None
        add_198: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_197, mul_265);  add_197 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_394: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_198, [4096, 16384]);  add_198 = None
        mm_54: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_394, permute_147)
        permute_280: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_55: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
        sum_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        view_395: "f32[16384]" = torch.ops.aten.reshape.default(sum_88, [16384]);  sum_88 = None
        add_199: "f32[16384]" = torch.ops.aten.add.Tensor(add_177, view_395);  add_177 = view_395 = None
        add_200: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_178, mm_55);  add_178 = mm_55 = None
        view_396: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_54, [8, 512, 4096]);  mm_54 = None
        add_201: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_255, view_396);  mul_255 = view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_267: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_201, primals_19)
        mul_268: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_267, 4096)
        sum_89: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_267, mul_74);  mul_267 = None
        sum_90: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_74, sum_90);  sum_90 = None
        sub_78: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_268, sum_89);  mul_268 = sum_89 = None
        sub_79: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_78, mul_270);  sub_78 = mul_270 = None
        mul_271: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_24, sub_79);  div_24 = sub_79 = None
        mul_272: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_201, mul_74);  mul_74 = None
        sum_91: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_92: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None
        add_202: "f32[4096]" = torch.ops.aten.add.Tensor(add_180, sum_91);  add_180 = sum_91 = None
        add_203: "f32[4096]" = torch.ops.aten.add.Tensor(add_181, sum_92);  add_181 = sum_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_397: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_271, [4096, 4096])
        mm_56: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_397, permute_151)
        permute_284: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_397, [1, 0])
        mm_57: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
        sum_93: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        view_398: "f32[4096]" = torch.ops.aten.reshape.default(sum_93, [4096]);  sum_93 = None
        add_204: "f32[4096]" = torch.ops.aten.add.Tensor(add_182, view_398);  add_182 = view_398 = None
        add_205: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_183, mm_57);  add_183 = mm_57 = None
        view_399: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_56, [8, 512, 4096]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_400: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_399, [8, 512, 64, 64]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_77: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_401: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_77, [512, 512, 64]);  clone_77 = None
        expand_33: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_15, [8, 64, 512, 512])
        view_168: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [512, 512, 512]);  expand_33 = None
        permute_288: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        bmm_40: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_288, view_401);  permute_288 = None
        bmm_41: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_401, permute_289);  view_401 = permute_289 = None
        view_402: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_40, [8, 64, 512, 64]);  bmm_40 = None
        view_403: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_41, [8, 64, 512, 512]);  bmm_41 = None
        mul_273: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_403, where_15);  view_403 = None
        sum_94: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [-1], True)
        neg_5: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_15);  where_15 = None
        fma_4: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_273);  neg_5 = sum_94 = mul_273 = None
        view_404: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_4, [512, 512, 512]);  fma_4 = None
        bmm_42: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_290, view_404);  permute_290 = None
        bmm_43: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_404, permute_291);  view_404 = permute_291 = None
        view_405: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_42, [8, 64, 64, 512]);  bmm_42 = None
        view_406: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [8, 64, 512, 64]);  bmm_43 = None
        mul_274: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_405, 0.3535533905932738);  view_405 = None
        permute_292: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_274, [0, 1, 3, 2]);  mul_274 = None
        mul_275: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_406, 0.3535533905932738);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_293: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_78: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_407: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_78, [8, 512, 4096]);  clone_78 = None
        view_408: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_407, [4096, 4096]);  view_407 = None
        mm_58: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_408, permute_162)
        permute_295: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_59: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_295, view_156);  permute_295 = None
        sum_95: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        view_409: "f32[4096]" = torch.ops.aten.reshape.default(sum_95, [4096]);  sum_95 = None
        add_206: "f32[4096]" = torch.ops.aten.add.Tensor(add_184, view_409);  add_184 = view_409 = None
        add_207: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_185, mm_59);  add_185 = mm_59 = None
        view_410: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_58, [8, 512, 4096]);  mm_58 = None
        add_208: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_271, view_410);  mul_271 = view_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_298: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
        view_411: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_298, [8, 512, 4096]);  permute_298 = None
        clone_79: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_411, memory_format = torch.contiguous_format);  view_411 = None
        view_412: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_79, [4096, 4096]);  clone_79 = None
        mm_60: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_412, permute_167)
        permute_300: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_61: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = None
        sum_96: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        view_413: "f32[4096]" = torch.ops.aten.reshape.default(sum_96, [4096]);  sum_96 = None
        add_209: "f32[4096]" = torch.ops.aten.add.Tensor(add_187, view_413);  add_187 = view_413 = None
        add_210: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_188, mm_61);  add_188 = mm_61 = None
        view_414: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_60, [8, 512, 4096]);  mm_60 = None
        add_211: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_208, view_414);  add_208 = view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_303: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_275, [0, 2, 1, 3]);  mul_275 = None
        clone_80: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_415: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_80, [8, 512, 4096]);  clone_80 = None
        view_416: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_415, [4096, 4096]);  view_415 = None
        mm_62: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_416, permute_172)
        permute_305: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_63: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
        sum_97: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        view_417: "f32[4096]" = torch.ops.aten.reshape.default(sum_97, [4096]);  sum_97 = None
        add_212: "f32[4096]" = torch.ops.aten.add.Tensor(add_190, view_417);  add_190 = view_417 = None
        add_213: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_191, mm_63);  add_191 = mm_63 = None
        view_418: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_62, [8, 512, 4096]);  mm_62 = None
        add_214: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_211, view_418);  add_211 = view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_277: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_214, primals_25)
        mul_278: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_277, 4096)
        sum_98: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
        mul_279: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_277, mul_70);  mul_277 = None
        sum_99: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        mul_280: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_70, sum_99);  sum_99 = None
        sub_81: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_278, sum_98);  mul_278 = sum_98 = None
        sub_82: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_81, mul_280);  sub_81 = mul_280 = None
        mul_281: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_25, sub_82);  div_25 = sub_82 = None
        mul_282: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_214, mul_70);  mul_70 = None
        sum_100: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
        sum_101: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_214, [0, 1]);  add_214 = None
        add_215: "f32[4096]" = torch.ops.aten.add.Tensor(add_193, sum_100);  add_193 = sum_100 = None
        add_216: "f32[4096]" = torch.ops.aten.add.Tensor(add_194, sum_101);  add_194 = sum_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_419: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_281, [4096, 4096])
        mm_64: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_419, permute_143)
        permute_309: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_419, [1, 0])
        mm_65: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
        sum_102: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        view_420: "f32[4096]" = torch.ops.aten.reshape.default(sum_102, [4096]);  sum_102 = None
        add_217: "f32[4096]" = torch.ops.aten.add.Tensor(add_195, view_420);  add_195 = view_420 = None
        add_218: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_196, mm_65);  add_196 = mm_65 = None
        view_421: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_64, [8, 512, 16384]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_153: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_41, [8, 512, 16384]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_283: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_421, mul_66);  mul_66 = None
        pow_7: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
        mul_67: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_153, mul_67);  mul_67 = None
        mul_68: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_65: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_284: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_421, add_65);  view_421 = add_65 = None
        mul_285: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_83: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_285);  mul_285 = None
        mul_286: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_283, sub_83);  mul_283 = sub_83 = None
        mul_287: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_286, 0.7978845608028654);  mul_286 = None
        mul_288: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_287, 0.044715)
        pow_20: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 2.0);  view_153 = None
        mul_289: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_290: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        add_219: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_287, mul_290);  mul_287 = mul_290 = None
        mul_291: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_284, 0.5);  mul_284 = None
        add_220: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_219, mul_291);  add_219 = mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_422: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_220, [4096, 16384]);  add_220 = None
        mm_66: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_422, permute_147)
        permute_313: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_67: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
        sum_103: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        view_423: "f32[16384]" = torch.ops.aten.reshape.default(sum_103, [16384]);  sum_103 = None
        add_221: "f32[16384]" = torch.ops.aten.add.Tensor(add_199, view_423);  add_199 = view_423 = None
        add_222: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_200, mm_67);  add_200 = mm_67 = None
        view_424: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_66, [8, 512, 4096]);  mm_66 = None
        add_223: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_281, view_424);  mul_281 = view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_293: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_223, primals_19)
        mul_294: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_293, 4096)
        sum_104: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True)
        mul_295: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_293, mul_64);  mul_293 = None
        sum_105: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True);  mul_295 = None
        mul_296: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_64, sum_105);  sum_105 = None
        sub_85: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_294, sum_104);  mul_294 = sum_104 = None
        sub_86: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_85, mul_296);  sub_85 = mul_296 = None
        mul_297: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_26, sub_86);  div_26 = sub_86 = None
        mul_298: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_223, mul_64);  mul_64 = None
        sum_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 1]);  mul_298 = None
        sum_107: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None
        add_224: "f32[4096]" = torch.ops.aten.add.Tensor(add_202, sum_106);  add_202 = sum_106 = None
        add_225: "f32[4096]" = torch.ops.aten.add.Tensor(add_203, sum_107);  add_203 = sum_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_425: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_297, [4096, 4096])
        mm_68: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_425, permute_151)
        permute_317: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_69: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
        sum_108: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        view_426: "f32[4096]" = torch.ops.aten.reshape.default(sum_108, [4096]);  sum_108 = None
        add_226: "f32[4096]" = torch.ops.aten.add.Tensor(add_204, view_426);  add_204 = view_426 = None
        add_227: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_205, mm_69);  add_205 = mm_69 = None
        view_427: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_68, [8, 512, 4096]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_428: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_427, [8, 512, 64, 64]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_320: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_81: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_429: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_81, [512, 512, 64]);  clone_81 = None
        expand_29: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_13, [8, 64, 512, 512])
        view_146: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [512, 512, 512]);  expand_29 = None
        permute_321: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
        bmm_44: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_321, view_429);  permute_321 = None
        bmm_45: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_429, permute_322);  view_429 = permute_322 = None
        view_430: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_44, [8, 64, 512, 64]);  bmm_44 = None
        view_431: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_45, [8, 64, 512, 512]);  bmm_45 = None
        mul_299: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_431, where_13);  view_431 = None
        sum_109: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [-1], True)
        neg_6: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_13);  where_13 = None
        fma_5: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_299);  neg_6 = sum_109 = mul_299 = None
        view_432: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_5, [512, 512, 512]);  fma_5 = None
        bmm_46: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_323, view_432);  permute_323 = None
        bmm_47: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_432, permute_324);  view_432 = permute_324 = None
        view_433: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_46, [8, 64, 64, 512]);  bmm_46 = None
        view_434: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [8, 64, 512, 64]);  bmm_47 = None
        mul_300: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_433, 0.3535533905932738);  view_433 = None
        permute_325: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_300, [0, 1, 3, 2]);  mul_300 = None
        mul_301: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_434, 0.3535533905932738);  view_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_326: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_82: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        view_435: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_82, [8, 512, 4096]);  clone_82 = None
        view_436: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_435, [4096, 4096]);  view_435 = None
        mm_70: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_436, permute_162)
        permute_328: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_71: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_328, view_134);  permute_328 = None
        sum_110: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        view_437: "f32[4096]" = torch.ops.aten.reshape.default(sum_110, [4096]);  sum_110 = None
        add_228: "f32[4096]" = torch.ops.aten.add.Tensor(add_206, view_437);  add_206 = view_437 = None
        add_229: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_207, mm_71);  add_207 = mm_71 = None
        view_438: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_70, [8, 512, 4096]);  mm_70 = None
        add_230: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_297, view_438);  mul_297 = view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_331: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
        view_439: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_331, [8, 512, 4096]);  permute_331 = None
        clone_83: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_439, memory_format = torch.contiguous_format);  view_439 = None
        view_440: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_83, [4096, 4096]);  clone_83 = None
        mm_72: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_440, permute_167)
        permute_333: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_440, [1, 0])
        mm_73: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = None
        sum_111: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
        view_441: "f32[4096]" = torch.ops.aten.reshape.default(sum_111, [4096]);  sum_111 = None
        add_231: "f32[4096]" = torch.ops.aten.add.Tensor(add_209, view_441);  add_209 = view_441 = None
        add_232: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_210, mm_73);  add_210 = mm_73 = None
        view_442: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_72, [8, 512, 4096]);  mm_72 = None
        add_233: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_230, view_442);  add_230 = view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_336: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_301, [0, 2, 1, 3]);  mul_301 = None
        clone_84: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_443: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_84, [8, 512, 4096]);  clone_84 = None
        view_444: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_443, [4096, 4096]);  view_443 = None
        mm_74: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_444, permute_172)
        permute_338: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_75: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
        sum_112: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        view_445: "f32[4096]" = torch.ops.aten.reshape.default(sum_112, [4096]);  sum_112 = None
        add_234: "f32[4096]" = torch.ops.aten.add.Tensor(add_212, view_445);  add_212 = view_445 = None
        add_235: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_213, mm_75);  add_213 = mm_75 = None
        view_446: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_74, [8, 512, 4096]);  mm_74 = None
        add_236: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_233, view_446);  add_233 = view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_303: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_236, primals_25)
        mul_304: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_303, 4096)
        sum_113: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
        mul_305: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_303, mul_60);  mul_303 = None
        sum_114: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
        mul_306: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_60, sum_114);  sum_114 = None
        sub_88: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_304, sum_113);  mul_304 = sum_113 = None
        sub_89: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_88, mul_306);  sub_88 = mul_306 = None
        mul_307: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_27, sub_89);  div_27 = sub_89 = None
        mul_308: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_236, mul_60);  mul_60 = None
        sum_115: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
        sum_116: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
        add_237: "f32[4096]" = torch.ops.aten.add.Tensor(add_215, sum_115);  add_215 = sum_115 = None
        add_238: "f32[4096]" = torch.ops.aten.add.Tensor(add_216, sum_116);  add_216 = sum_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_447: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_307, [4096, 4096])
        mm_76: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_447, permute_143)
        permute_342: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_447, [1, 0])
        mm_77: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
        sum_117: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        view_448: "f32[4096]" = torch.ops.aten.reshape.default(sum_117, [4096]);  sum_117 = None
        add_239: "f32[4096]" = torch.ops.aten.add.Tensor(add_217, view_448);  add_217 = view_448 = None
        add_240: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_218, mm_77);  add_218 = mm_77 = None
        view_449: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_76, [8, 512, 16384]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_131: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_35, [8, 512, 16384]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_309: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_449, mul_56);  mul_56 = None
        pow_6: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
        mul_57: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_131, mul_57);  mul_57 = None
        mul_58: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_56: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_310: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_449, add_56);  view_449 = add_56 = None
        mul_311: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_90: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_311);  mul_311 = None
        mul_312: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_309, sub_90);  mul_309 = sub_90 = None
        mul_313: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_312, 0.7978845608028654);  mul_312 = None
        mul_314: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_313, 0.044715)
        pow_21: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
        mul_315: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_316: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
        add_241: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_313, mul_316);  mul_313 = mul_316 = None
        mul_317: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_310, 0.5);  mul_310 = None
        add_242: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_241, mul_317);  add_241 = mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_450: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_242, [4096, 16384]);  add_242 = None
        mm_78: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_450, permute_147)
        permute_346: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_450, [1, 0])
        mm_79: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
        sum_118: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        view_451: "f32[16384]" = torch.ops.aten.reshape.default(sum_118, [16384]);  sum_118 = None
        add_243: "f32[16384]" = torch.ops.aten.add.Tensor(add_221, view_451);  add_221 = view_451 = None
        add_244: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_222, mm_79);  add_222 = mm_79 = None
        view_452: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_78, [8, 512, 4096]);  mm_78 = None
        add_245: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_307, view_452);  mul_307 = view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_319: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_245, primals_19)
        mul_320: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_319, 4096)
        sum_119: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
        mul_321: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_319, mul_54);  mul_319 = None
        sum_120: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
        mul_322: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_54, sum_120);  sum_120 = None
        sub_92: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_320, sum_119);  mul_320 = sum_119 = None
        sub_93: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_92, mul_322);  sub_92 = mul_322 = None
        mul_323: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_28, sub_93);  div_28 = sub_93 = None
        mul_324: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_245, mul_54);  mul_54 = None
        sum_121: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
        sum_122: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        add_246: "f32[4096]" = torch.ops.aten.add.Tensor(add_224, sum_121);  add_224 = sum_121 = None
        add_247: "f32[4096]" = torch.ops.aten.add.Tensor(add_225, sum_122);  add_225 = sum_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_453: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_323, [4096, 4096])
        mm_80: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_453, permute_151)
        permute_350: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_81: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
        sum_123: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        view_454: "f32[4096]" = torch.ops.aten.reshape.default(sum_123, [4096]);  sum_123 = None
        add_248: "f32[4096]" = torch.ops.aten.add.Tensor(add_226, view_454);  add_226 = view_454 = None
        add_249: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_227, mm_81);  add_227 = mm_81 = None
        view_455: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_80, [8, 512, 4096]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_456: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_455, [8, 512, 64, 64]);  view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_353: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_85: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_457: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_85, [512, 512, 64]);  clone_85 = None
        expand_25: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_11, [8, 64, 512, 512])
        view_124: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [512, 512, 512]);  expand_25 = None
        permute_354: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        bmm_48: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_354, view_457);  permute_354 = None
        bmm_49: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_457, permute_355);  view_457 = permute_355 = None
        view_458: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [8, 64, 512, 64]);  bmm_48 = None
        view_459: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [8, 64, 512, 512]);  bmm_49 = None
        mul_325: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_459, where_11);  view_459 = None
        sum_124: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [-1], True)
        neg_7: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_11);  where_11 = None
        fma_6: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_325);  neg_7 = sum_124 = mul_325 = None
        view_460: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_6, [512, 512, 512]);  fma_6 = None
        bmm_50: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_356, view_460);  permute_356 = None
        bmm_51: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_460, permute_357);  view_460 = permute_357 = None
        view_461: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_50, [8, 64, 64, 512]);  bmm_50 = None
        view_462: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [8, 64, 512, 64]);  bmm_51 = None
        mul_326: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_461, 0.3535533905932738);  view_461 = None
        permute_358: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_326, [0, 1, 3, 2]);  mul_326 = None
        mul_327: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_462, 0.3535533905932738);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_359: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_86: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_463: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_86, [8, 512, 4096]);  clone_86 = None
        view_464: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_463, [4096, 4096]);  view_463 = None
        mm_82: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_464, permute_162)
        permute_361: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_83: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_361, view_112);  permute_361 = None
        sum_125: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        view_465: "f32[4096]" = torch.ops.aten.reshape.default(sum_125, [4096]);  sum_125 = None
        add_250: "f32[4096]" = torch.ops.aten.add.Tensor(add_228, view_465);  add_228 = view_465 = None
        add_251: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_229, mm_83);  add_229 = mm_83 = None
        view_466: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_82, [8, 512, 4096]);  mm_82 = None
        add_252: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_323, view_466);  mul_323 = view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_364: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
        view_467: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_364, [8, 512, 4096]);  permute_364 = None
        clone_87: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_467, memory_format = torch.contiguous_format);  view_467 = None
        view_468: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_87, [4096, 4096]);  clone_87 = None
        mm_84: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_468, permute_167)
        permute_366: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_85: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = None
        sum_126: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        view_469: "f32[4096]" = torch.ops.aten.reshape.default(sum_126, [4096]);  sum_126 = None
        add_253: "f32[4096]" = torch.ops.aten.add.Tensor(add_231, view_469);  add_231 = view_469 = None
        add_254: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_232, mm_85);  add_232 = mm_85 = None
        view_470: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_84, [8, 512, 4096]);  mm_84 = None
        add_255: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_252, view_470);  add_252 = view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_369: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_327, [0, 2, 1, 3]);  mul_327 = None
        clone_88: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_471: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_88, [8, 512, 4096]);  clone_88 = None
        view_472: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_471, [4096, 4096]);  view_471 = None
        mm_86: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_472, permute_172)
        permute_371: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_87: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
        sum_127: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        view_473: "f32[4096]" = torch.ops.aten.reshape.default(sum_127, [4096]);  sum_127 = None
        add_256: "f32[4096]" = torch.ops.aten.add.Tensor(add_234, view_473);  add_234 = view_473 = None
        add_257: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_235, mm_87);  add_235 = mm_87 = None
        view_474: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_86, [8, 512, 4096]);  mm_86 = None
        add_258: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_255, view_474);  add_255 = view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_329: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_258, primals_25)
        mul_330: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_329, 4096)
        sum_128: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
        mul_331: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_329, mul_50);  mul_329 = None
        sum_129: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
        mul_332: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_50, sum_129);  sum_129 = None
        sub_95: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_330, sum_128);  mul_330 = sum_128 = None
        sub_96: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_95, mul_332);  sub_95 = mul_332 = None
        mul_333: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_29, sub_96);  div_29 = sub_96 = None
        mul_334: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_258, mul_50);  mul_50 = None
        sum_130: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
        sum_131: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_258, [0, 1]);  add_258 = None
        add_259: "f32[4096]" = torch.ops.aten.add.Tensor(add_237, sum_130);  add_237 = sum_130 = None
        add_260: "f32[4096]" = torch.ops.aten.add.Tensor(add_238, sum_131);  add_238 = sum_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_475: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_333, [4096, 4096])
        mm_88: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_475, permute_143)
        permute_375: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_89: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
        sum_132: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        view_476: "f32[4096]" = torch.ops.aten.reshape.default(sum_132, [4096]);  sum_132 = None
        add_261: "f32[4096]" = torch.ops.aten.add.Tensor(add_239, view_476);  add_239 = view_476 = None
        add_262: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_240, mm_89);  add_240 = mm_89 = None
        view_477: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_88, [8, 512, 16384]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_109: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_29, [8, 512, 16384]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_335: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_477, mul_46);  mul_46 = None
        pow_5: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
        mul_47: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_109, mul_47);  mul_47 = None
        mul_48: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_47: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_336: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_477, add_47);  view_477 = add_47 = None
        mul_337: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_97: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_337);  mul_337 = None
        mul_338: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_335, sub_97);  mul_335 = sub_97 = None
        mul_339: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_338, 0.7978845608028654);  mul_338 = None
        mul_340: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_339, 0.044715)
        pow_22: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
        mul_341: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_342: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        add_263: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_339, mul_342);  mul_339 = mul_342 = None
        mul_343: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_336, 0.5);  mul_336 = None
        add_264: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_263, mul_343);  add_263 = mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_478: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_264, [4096, 16384]);  add_264 = None
        mm_90: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_478, permute_147)
        permute_379: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_91: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
        sum_133: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        view_479: "f32[16384]" = torch.ops.aten.reshape.default(sum_133, [16384]);  sum_133 = None
        add_265: "f32[16384]" = torch.ops.aten.add.Tensor(add_243, view_479);  add_243 = view_479 = None
        add_266: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_244, mm_91);  add_244 = mm_91 = None
        view_480: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_90, [8, 512, 4096]);  mm_90 = None
        add_267: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_333, view_480);  mul_333 = view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_345: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_267, primals_19)
        mul_346: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_345, 4096)
        sum_134: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
        mul_347: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_345, mul_44);  mul_345 = None
        sum_135: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_44, sum_135);  sum_135 = None
        sub_99: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_346, sum_134);  mul_346 = sum_134 = None
        sub_100: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_99, mul_348);  sub_99 = mul_348 = None
        mul_349: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_30, sub_100);  div_30 = sub_100 = None
        mul_350: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_267, mul_44);  mul_44 = None
        sum_136: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
        sum_137: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
        add_268: "f32[4096]" = torch.ops.aten.add.Tensor(add_246, sum_136);  add_246 = sum_136 = None
        add_269: "f32[4096]" = torch.ops.aten.add.Tensor(add_247, sum_137);  add_247 = sum_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_481: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_349, [4096, 4096])
        mm_92: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_481, permute_151)
        permute_383: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_93: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
        sum_138: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        view_482: "f32[4096]" = torch.ops.aten.reshape.default(sum_138, [4096]);  sum_138 = None
        add_270: "f32[4096]" = torch.ops.aten.add.Tensor(add_248, view_482);  add_248 = view_482 = None
        add_271: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_249, mm_93);  add_249 = mm_93 = None
        view_483: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_92, [8, 512, 4096]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_484: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_483, [8, 512, 64, 64]);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_386: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_89: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_485: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_89, [512, 512, 64]);  clone_89 = None
        expand_21: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_9, [8, 64, 512, 512])
        view_102: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [512, 512, 512]);  expand_21 = None
        permute_387: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        bmm_52: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_387, view_485);  permute_387 = None
        bmm_53: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_485, permute_388);  view_485 = permute_388 = None
        view_486: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [8, 64, 512, 64]);  bmm_52 = None
        view_487: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_53, [8, 64, 512, 512]);  bmm_53 = None
        mul_351: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_487, where_9);  view_487 = None
        sum_139: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
        neg_8: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_9);  where_9 = None
        fma_7: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_351);  neg_8 = sum_139 = mul_351 = None
        view_488: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_7, [512, 512, 512]);  fma_7 = None
        bmm_54: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_389, view_488);  permute_389 = None
        bmm_55: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_488, permute_390);  view_488 = permute_390 = None
        view_489: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_54, [8, 64, 64, 512]);  bmm_54 = None
        view_490: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [8, 64, 512, 64]);  bmm_55 = None
        mul_352: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_489, 0.3535533905932738);  view_489 = None
        permute_391: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_352, [0, 1, 3, 2]);  mul_352 = None
        mul_353: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_490, 0.3535533905932738);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_392: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_90: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_491: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_90, [8, 512, 4096]);  clone_90 = None
        view_492: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_491, [4096, 4096]);  view_491 = None
        mm_94: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_492, permute_162)
        permute_394: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_95: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_394, view_90);  permute_394 = None
        sum_140: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        view_493: "f32[4096]" = torch.ops.aten.reshape.default(sum_140, [4096]);  sum_140 = None
        add_272: "f32[4096]" = torch.ops.aten.add.Tensor(add_250, view_493);  add_250 = view_493 = None
        add_273: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_251, mm_95);  add_251 = mm_95 = None
        view_494: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_94, [8, 512, 4096]);  mm_94 = None
        add_274: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_349, view_494);  mul_349 = view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_397: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
        view_495: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_397, [8, 512, 4096]);  permute_397 = None
        clone_91: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_495, memory_format = torch.contiguous_format);  view_495 = None
        view_496: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_91, [4096, 4096]);  clone_91 = None
        mm_96: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_496, permute_167)
        permute_399: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_97: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = None
        sum_141: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        view_497: "f32[4096]" = torch.ops.aten.reshape.default(sum_141, [4096]);  sum_141 = None
        add_275: "f32[4096]" = torch.ops.aten.add.Tensor(add_253, view_497);  add_253 = view_497 = None
        add_276: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_254, mm_97);  add_254 = mm_97 = None
        view_498: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_96, [8, 512, 4096]);  mm_96 = None
        add_277: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_274, view_498);  add_274 = view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_402: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_353, [0, 2, 1, 3]);  mul_353 = None
        clone_92: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
        view_499: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_92, [8, 512, 4096]);  clone_92 = None
        view_500: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_499, [4096, 4096]);  view_499 = None
        mm_98: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_500, permute_172)
        permute_404: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_500, [1, 0])
        mm_99: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
        sum_142: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        view_501: "f32[4096]" = torch.ops.aten.reshape.default(sum_142, [4096]);  sum_142 = None
        add_278: "f32[4096]" = torch.ops.aten.add.Tensor(add_256, view_501);  add_256 = view_501 = None
        add_279: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_257, mm_99);  add_257 = mm_99 = None
        view_502: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_98, [8, 512, 4096]);  mm_98 = None
        add_280: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_277, view_502);  add_277 = view_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_355: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_280, primals_25)
        mul_356: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_355, 4096)
        sum_143: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
        mul_357: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_355, mul_40);  mul_355 = None
        sum_144: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
        mul_358: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, sum_144);  sum_144 = None
        sub_102: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_356, sum_143);  mul_356 = sum_143 = None
        sub_103: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_102, mul_358);  sub_102 = mul_358 = None
        mul_359: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_31, sub_103);  div_31 = sub_103 = None
        mul_360: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_280, mul_40);  mul_40 = None
        sum_145: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
        sum_146: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
        add_281: "f32[4096]" = torch.ops.aten.add.Tensor(add_259, sum_145);  add_259 = sum_145 = None
        add_282: "f32[4096]" = torch.ops.aten.add.Tensor(add_260, sum_146);  add_260 = sum_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_503: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_359, [4096, 4096])
        mm_100: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_503, permute_143)
        permute_408: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_503, [1, 0])
        mm_101: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
        sum_147: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        view_504: "f32[4096]" = torch.ops.aten.reshape.default(sum_147, [4096]);  sum_147 = None
        add_283: "f32[4096]" = torch.ops.aten.add.Tensor(add_261, view_504);  add_261 = view_504 = None
        add_284: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_262, mm_101);  add_262 = mm_101 = None
        view_505: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_100, [8, 512, 16384]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_87: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_23, [8, 512, 16384]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_361: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_505, mul_36);  mul_36 = None
        pow_4: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
        mul_37: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_87, mul_37);  mul_37 = None
        mul_38: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_38: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_362: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_505, add_38);  view_505 = add_38 = None
        mul_363: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_104: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_363);  mul_363 = None
        mul_364: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_361, sub_104);  mul_361 = sub_104 = None
        mul_365: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_364, 0.7978845608028654);  mul_364 = None
        mul_366: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_365, 0.044715)
        pow_23: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 2.0);  view_87 = None
        mul_367: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_368: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
        add_285: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_365, mul_368);  mul_365 = mul_368 = None
        mul_369: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_362, 0.5);  mul_362 = None
        add_286: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_285, mul_369);  add_285 = mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_506: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_286, [4096, 16384]);  add_286 = None
        mm_102: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_506, permute_147)
        permute_412: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_506, [1, 0])
        mm_103: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
        sum_148: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        view_507: "f32[16384]" = torch.ops.aten.reshape.default(sum_148, [16384]);  sum_148 = None
        add_287: "f32[16384]" = torch.ops.aten.add.Tensor(add_265, view_507);  add_265 = view_507 = None
        add_288: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_266, mm_103);  add_266 = mm_103 = None
        view_508: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_102, [8, 512, 4096]);  mm_102 = None
        add_289: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_359, view_508);  mul_359 = view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_371: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_289, primals_19)
        mul_372: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_371, 4096)
        sum_149: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
        mul_373: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_371, mul_34);  mul_371 = None
        sum_150: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
        mul_374: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_34, sum_150);  sum_150 = None
        sub_106: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_372, sum_149);  mul_372 = sum_149 = None
        sub_107: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_106, mul_374);  sub_106 = mul_374 = None
        mul_375: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_32, sub_107);  div_32 = sub_107 = None
        mul_376: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_289, mul_34);  mul_34 = None
        sum_151: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
        sum_152: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_289, [0, 1]);  add_289 = None
        add_290: "f32[4096]" = torch.ops.aten.add.Tensor(add_268, sum_151);  add_268 = sum_151 = None
        add_291: "f32[4096]" = torch.ops.aten.add.Tensor(add_269, sum_152);  add_269 = sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_509: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_375, [4096, 4096])
        mm_104: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_509, permute_151)
        permute_416: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_509, [1, 0])
        mm_105: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
        sum_153: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        view_510: "f32[4096]" = torch.ops.aten.reshape.default(sum_153, [4096]);  sum_153 = None
        add_292: "f32[4096]" = torch.ops.aten.add.Tensor(add_270, view_510);  add_270 = view_510 = None
        add_293: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_271, mm_105);  add_271 = mm_105 = None
        view_511: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_104, [8, 512, 4096]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_512: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_511, [8, 512, 64, 64]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_419: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_93: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_513: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_93, [512, 512, 64]);  clone_93 = None
        expand_17: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_7, [8, 64, 512, 512])
        view_80: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [512, 512, 512]);  expand_17 = None
        permute_420: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        bmm_56: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_420, view_513);  permute_420 = None
        bmm_57: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_513, permute_421);  view_513 = permute_421 = None
        view_514: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [8, 64, 512, 64]);  bmm_56 = None
        view_515: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_57, [8, 64, 512, 512]);  bmm_57 = None
        mul_377: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_515, where_7);  view_515 = None
        sum_154: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [-1], True)
        neg_9: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_7);  where_7 = None
        fma_8: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_377);  neg_9 = sum_154 = mul_377 = None
        view_516: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_8, [512, 512, 512]);  fma_8 = None
        bmm_58: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_422, view_516);  permute_422 = None
        bmm_59: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_516, permute_423);  view_516 = permute_423 = None
        view_517: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_58, [8, 64, 64, 512]);  bmm_58 = None
        view_518: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [8, 64, 512, 64]);  bmm_59 = None
        mul_378: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_517, 0.3535533905932738);  view_517 = None
        permute_424: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_378, [0, 1, 3, 2]);  mul_378 = None
        mul_379: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_518, 0.3535533905932738);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_425: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        clone_94: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
        view_519: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_94, [8, 512, 4096]);  clone_94 = None
        view_520: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_519, [4096, 4096]);  view_519 = None
        mm_106: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_520, permute_162)
        permute_427: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_107: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_427, view_68);  permute_427 = None
        sum_155: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        view_521: "f32[4096]" = torch.ops.aten.reshape.default(sum_155, [4096]);  sum_155 = None
        add_294: "f32[4096]" = torch.ops.aten.add.Tensor(add_272, view_521);  add_272 = view_521 = None
        add_295: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_273, mm_107);  add_273 = mm_107 = None
        view_522: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_106, [8, 512, 4096]);  mm_106 = None
        add_296: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_375, view_522);  mul_375 = view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_430: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
        view_523: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_430, [8, 512, 4096]);  permute_430 = None
        clone_95: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_523, memory_format = torch.contiguous_format);  view_523 = None
        view_524: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_95, [4096, 4096]);  clone_95 = None
        mm_108: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_524, permute_167)
        permute_432: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_524, [1, 0])
        mm_109: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = None
        sum_156: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
        view_525: "f32[4096]" = torch.ops.aten.reshape.default(sum_156, [4096]);  sum_156 = None
        add_297: "f32[4096]" = torch.ops.aten.add.Tensor(add_275, view_525);  add_275 = view_525 = None
        add_298: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_276, mm_109);  add_276 = mm_109 = None
        view_526: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_108, [8, 512, 4096]);  mm_108 = None
        add_299: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_296, view_526);  add_296 = view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_435: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_379, [0, 2, 1, 3]);  mul_379 = None
        clone_96: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
        view_527: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_96, [8, 512, 4096]);  clone_96 = None
        view_528: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_527, [4096, 4096]);  view_527 = None
        mm_110: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_528, permute_172)
        permute_437: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_528, [1, 0])
        mm_111: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
        sum_157: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        view_529: "f32[4096]" = torch.ops.aten.reshape.default(sum_157, [4096]);  sum_157 = None
        add_300: "f32[4096]" = torch.ops.aten.add.Tensor(add_278, view_529);  add_278 = view_529 = None
        add_301: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_279, mm_111);  add_279 = mm_111 = None
        view_530: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_110, [8, 512, 4096]);  mm_110 = None
        add_302: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_299, view_530);  add_299 = view_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_381: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_302, primals_25)
        mul_382: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_381, 4096)
        sum_158: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
        mul_383: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_381, mul_30);  mul_381 = None
        sum_159: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
        mul_384: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_30, sum_159);  sum_159 = None
        sub_109: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_382, sum_158);  mul_382 = sum_158 = None
        sub_110: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_109, mul_384);  sub_109 = mul_384 = None
        mul_385: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_33, sub_110);  div_33 = sub_110 = None
        mul_386: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_302, mul_30);  mul_30 = None
        sum_160: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
        sum_161: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
        add_303: "f32[4096]" = torch.ops.aten.add.Tensor(add_281, sum_160);  add_281 = sum_160 = None
        add_304: "f32[4096]" = torch.ops.aten.add.Tensor(add_282, sum_161);  add_282 = sum_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_531: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_385, [4096, 4096])
        mm_112: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_531, permute_143)
        permute_441: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_531, [1, 0])
        mm_113: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
        sum_162: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        view_532: "f32[4096]" = torch.ops.aten.reshape.default(sum_162, [4096]);  sum_162 = None
        add_305: "f32[4096]" = torch.ops.aten.add.Tensor(add_283, view_532);  add_283 = view_532 = None
        add_306: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_284, mm_113);  add_284 = mm_113 = None
        view_533: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_112, [8, 512, 16384]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_65: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_17, [8, 512, 16384]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_387: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_533, mul_26);  mul_26 = None
        pow_3: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_27: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_65, mul_27);  mul_27 = None
        mul_28: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_29: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_388: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_533, add_29);  view_533 = add_29 = None
        mul_389: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_111: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_389);  mul_389 = None
        mul_390: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_387, sub_111);  mul_387 = sub_111 = None
        mul_391: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_390, 0.7978845608028654);  mul_390 = None
        mul_392: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_391, 0.044715)
        pow_24: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 2.0);  view_65 = None
        mul_393: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_394: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        add_307: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_391, mul_394);  mul_391 = mul_394 = None
        mul_395: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_388, 0.5);  mul_388 = None
        add_308: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_307, mul_395);  add_307 = mul_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_534: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_308, [4096, 16384]);  add_308 = None
        mm_114: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_534, permute_147)
        permute_445: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_115: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
        sum_163: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        view_535: "f32[16384]" = torch.ops.aten.reshape.default(sum_163, [16384]);  sum_163 = None
        add_309: "f32[16384]" = torch.ops.aten.add.Tensor(add_287, view_535);  add_287 = view_535 = None
        add_310: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_288, mm_115);  add_288 = mm_115 = None
        view_536: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_114, [8, 512, 4096]);  mm_114 = None
        add_311: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_385, view_536);  mul_385 = view_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_397: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_311, primals_19)
        mul_398: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_397, 4096)
        sum_164: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
        mul_399: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_397, mul_24);  mul_397 = None
        sum_165: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
        mul_400: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_24, sum_165);  sum_165 = None
        sub_113: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_398, sum_164);  mul_398 = sum_164 = None
        sub_114: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_113, mul_400);  sub_113 = mul_400 = None
        mul_401: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_34, sub_114);  div_34 = sub_114 = None
        mul_402: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_311, mul_24);  mul_24 = None
        sum_166: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
        sum_167: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        add_312: "f32[4096]" = torch.ops.aten.add.Tensor(add_290, sum_166);  add_290 = sum_166 = None
        add_313: "f32[4096]" = torch.ops.aten.add.Tensor(add_291, sum_167);  add_291 = sum_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_537: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_401, [4096, 4096])
        mm_116: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_537, permute_151)
        permute_449: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_117: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
        sum_168: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        view_538: "f32[4096]" = torch.ops.aten.reshape.default(sum_168, [4096]);  sum_168 = None
        add_314: "f32[4096]" = torch.ops.aten.add.Tensor(add_292, view_538);  add_292 = view_538 = None
        add_315: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_293, mm_117);  add_293 = mm_117 = None
        view_539: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_116, [8, 512, 4096]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_540: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_539, [8, 512, 64, 64]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_452: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_97: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_541: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_97, [512, 512, 64]);  clone_97 = None
        expand_13: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_5, [8, 64, 512, 512])
        view_58: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [512, 512, 512]);  expand_13 = None
        permute_453: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        bmm_60: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_453, view_541);  permute_453 = None
        bmm_61: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_541, permute_454);  view_541 = permute_454 = None
        view_542: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [8, 64, 512, 64]);  bmm_60 = None
        view_543: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_61, [8, 64, 512, 512]);  bmm_61 = None
        mul_403: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_543, where_5);  view_543 = None
        sum_169: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [-1], True)
        neg_10: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_5);  where_5 = None
        fma_9: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_403);  neg_10 = sum_169 = mul_403 = None
        view_544: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_9, [512, 512, 512]);  fma_9 = None
        bmm_62: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_455, view_544);  permute_455 = None
        bmm_63: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_544, permute_456);  view_544 = permute_456 = None
        view_545: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_62, [8, 64, 64, 512]);  bmm_62 = None
        view_546: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [8, 64, 512, 64]);  bmm_63 = None
        mul_404: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_545, 0.3535533905932738);  view_545 = None
        permute_457: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_404, [0, 1, 3, 2]);  mul_404 = None
        mul_405: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_546, 0.3535533905932738);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_458: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
        clone_98: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_547: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_98, [8, 512, 4096]);  clone_98 = None
        view_548: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_547, [4096, 4096]);  view_547 = None
        mm_118: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_548, permute_162)
        permute_460: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_119: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_460, view_46);  permute_460 = None
        sum_170: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549: "f32[4096]" = torch.ops.aten.reshape.default(sum_170, [4096]);  sum_170 = None
        add_316: "f32[4096]" = torch.ops.aten.add.Tensor(add_294, view_549);  add_294 = view_549 = None
        add_317: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_295, mm_119);  add_295 = mm_119 = None
        view_550: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_118, [8, 512, 4096]);  mm_118 = None
        add_318: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_401, view_550);  mul_401 = view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_463: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
        view_551: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_463, [8, 512, 4096]);  permute_463 = None
        clone_99: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_551, memory_format = torch.contiguous_format);  view_551 = None
        view_552: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_99, [4096, 4096]);  clone_99 = None
        mm_120: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_552, permute_167)
        permute_465: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_121: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = None
        sum_171: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
        view_553: "f32[4096]" = torch.ops.aten.reshape.default(sum_171, [4096]);  sum_171 = None
        add_319: "f32[4096]" = torch.ops.aten.add.Tensor(add_297, view_553);  add_297 = view_553 = None
        add_320: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_298, mm_121);  add_298 = mm_121 = None
        view_554: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_120, [8, 512, 4096]);  mm_120 = None
        add_321: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_318, view_554);  add_318 = view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_468: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_405, [0, 2, 1, 3]);  mul_405 = None
        clone_100: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_555: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_100, [8, 512, 4096]);  clone_100 = None
        view_556: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_555, [4096, 4096]);  view_555 = None
        mm_122: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_556, permute_172)
        permute_470: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_123: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
        sum_172: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        view_557: "f32[4096]" = torch.ops.aten.reshape.default(sum_172, [4096]);  sum_172 = None
        add_322: "f32[4096]" = torch.ops.aten.add.Tensor(add_300, view_557);  add_300 = view_557 = None
        add_323: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_301, mm_123);  add_301 = mm_123 = None
        view_558: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_122, [8, 512, 4096]);  mm_122 = None
        add_324: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_321, view_558);  add_321 = view_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_407: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_324, primals_25)
        mul_408: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_407, 4096)
        sum_173: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True)
        mul_409: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_407, mul_20);  mul_407 = None
        sum_174: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True);  mul_409 = None
        mul_410: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_20, sum_174);  sum_174 = None
        sub_116: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_408, sum_173);  mul_408 = sum_173 = None
        sub_117: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_116, mul_410);  sub_116 = mul_410 = None
        mul_411: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_35, sub_117);  div_35 = sub_117 = None
        mul_412: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_324, mul_20);  mul_20 = None
        sum_175: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 1]);  mul_412 = None
        sum_176: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_324, [0, 1]);  add_324 = None
        add_325: "f32[4096]" = torch.ops.aten.add.Tensor(add_303, sum_175);  add_303 = sum_175 = None
        add_326: "f32[4096]" = torch.ops.aten.add.Tensor(add_304, sum_176);  add_304 = sum_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_559: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_411, [4096, 4096])
        mm_124: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_559, permute_143)
        permute_474: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_559, [1, 0])
        mm_125: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
        sum_177: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        view_560: "f32[4096]" = torch.ops.aten.reshape.default(sum_177, [4096]);  sum_177 = None
        add_327: "f32[4096]" = torch.ops.aten.add.Tensor(add_305, view_560);  add_305 = view_560 = None
        add_328: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_306, mm_125);  add_306 = mm_125 = None
        view_561: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_124, [8, 512, 16384]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_43: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_11, [8, 512, 16384]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_413: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_561, mul_16);  mul_16 = None
        pow_2: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
        mul_17: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_43, mul_17);  mul_17 = None
        mul_18: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_20: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_414: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_561, add_20);  view_561 = add_20 = None
        mul_415: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_118: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_415);  mul_415 = None
        mul_416: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_413, sub_118);  mul_413 = sub_118 = None
        mul_417: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_416, 0.7978845608028654);  mul_416 = None
        mul_418: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_417, 0.044715)
        pow_25: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
        mul_419: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
        mul_420: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
        add_329: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_417, mul_420);  mul_417 = mul_420 = None
        mul_421: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_414, 0.5);  mul_414 = None
        add_330: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_329, mul_421);  add_329 = mul_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_562: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_330, [4096, 16384]);  add_330 = None
        mm_126: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_562, permute_147)
        permute_478: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_127: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
        sum_178: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        view_563: "f32[16384]" = torch.ops.aten.reshape.default(sum_178, [16384]);  sum_178 = None
        add_331: "f32[16384]" = torch.ops.aten.add.Tensor(add_309, view_563);  add_309 = view_563 = None
        add_332: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_310, mm_127);  add_310 = mm_127 = None
        view_564: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_126, [8, 512, 4096]);  mm_126 = None
        add_333: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_411, view_564);  mul_411 = view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_423: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_333, primals_19)
        mul_424: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_423, 4096)
        sum_179: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True)
        mul_425: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_423, mul_14);  mul_423 = None
        sum_180: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
        mul_426: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_14, sum_180);  sum_180 = None
        sub_120: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_424, sum_179);  mul_424 = sum_179 = None
        sub_121: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_120, mul_426);  sub_120 = mul_426 = None
        mul_427: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_36, sub_121);  div_36 = sub_121 = None
        mul_428: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_333, mul_14);  mul_14 = None
        sum_181: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
        sum_182: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_333, [0, 1]);  add_333 = None
        add_334: "f32[4096]" = torch.ops.aten.add.Tensor(add_312, sum_181);  add_312 = sum_181 = None
        add_335: "f32[4096]" = torch.ops.aten.add.Tensor(add_313, sum_182);  add_313 = sum_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_565: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_427, [4096, 4096])
        mm_128: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_565, permute_151)
        permute_482: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_129: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
        sum_183: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        view_566: "f32[4096]" = torch.ops.aten.reshape.default(sum_183, [4096]);  sum_183 = None
        add_336: "f32[4096]" = torch.ops.aten.add.Tensor(add_314, view_566);  add_314 = view_566 = None
        add_337: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_315, mm_129);  add_315 = mm_129 = None
        view_567: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_128, [8, 512, 4096]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_568: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_567, [8, 512, 64, 64]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_485: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_101: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_569: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_101, [512, 512, 64]);  clone_101 = None
        expand_9: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_3, [8, 64, 512, 512])
        view_36: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [512, 512, 512]);  expand_9 = None
        permute_486: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        bmm_64: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_486, view_569);  permute_486 = None
        bmm_65: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_569, permute_487);  view_569 = permute_487 = None
        view_570: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [8, 64, 512, 64]);  bmm_64 = None
        view_571: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, [8, 64, 512, 512]);  bmm_65 = None
        mul_429: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_571, where_3);  view_571 = None
        sum_184: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [-1], True)
        neg_11: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_10: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_429);  neg_11 = sum_184 = mul_429 = None
        view_572: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_10, [512, 512, 512]);  fma_10 = None
        bmm_66: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_488, view_572);  permute_488 = None
        bmm_67: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_572, permute_489);  view_572 = permute_489 = None
        view_573: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, [8, 64, 64, 512]);  bmm_66 = None
        view_574: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [8, 64, 512, 64]);  bmm_67 = None
        mul_430: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_573, 0.3535533905932738);  view_573 = None
        permute_490: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_430, [0, 1, 3, 2]);  mul_430 = None
        mul_431: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_574, 0.3535533905932738);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_491: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        clone_102: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_575: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_102, [8, 512, 4096]);  clone_102 = None
        view_576: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_575, [4096, 4096]);  view_575 = None
        mm_130: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_576, permute_162)
        permute_493: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_131: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_493, view_24);  permute_493 = None
        sum_185: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        view_577: "f32[4096]" = torch.ops.aten.reshape.default(sum_185, [4096]);  sum_185 = None
        add_338: "f32[4096]" = torch.ops.aten.add.Tensor(add_316, view_577);  add_316 = view_577 = None
        add_339: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_317, mm_131);  add_317 = mm_131 = None
        view_578: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_130, [8, 512, 4096]);  mm_130 = None
        add_340: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_427, view_578);  mul_427 = view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_496: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
        view_579: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_496, [8, 512, 4096]);  permute_496 = None
        clone_103: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_579, memory_format = torch.contiguous_format);  view_579 = None
        view_580: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_103, [4096, 4096]);  clone_103 = None
        mm_132: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_580, permute_167)
        permute_498: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_580, [1, 0])
        mm_133: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = None
        sum_186: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        view_581: "f32[4096]" = torch.ops.aten.reshape.default(sum_186, [4096]);  sum_186 = None
        add_341: "f32[4096]" = torch.ops.aten.add.Tensor(add_319, view_581);  add_319 = view_581 = None
        add_342: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_320, mm_133);  add_320 = mm_133 = None
        view_582: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_132, [8, 512, 4096]);  mm_132 = None
        add_343: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_340, view_582);  add_340 = view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_501: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_431, [0, 2, 1, 3]);  mul_431 = None
        clone_104: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_583: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_104, [8, 512, 4096]);  clone_104 = None
        view_584: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_583, [4096, 4096]);  view_583 = None
        mm_134: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_584, permute_172)
        permute_503: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_584, [1, 0])
        mm_135: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
        sum_187: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        view_585: "f32[4096]" = torch.ops.aten.reshape.default(sum_187, [4096]);  sum_187 = None
        add_344: "f32[4096]" = torch.ops.aten.add.Tensor(add_322, view_585);  add_322 = view_585 = None
        add_345: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_323, mm_135);  add_323 = mm_135 = None
        view_586: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_134, [8, 512, 4096]);  mm_134 = None
        add_346: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_343, view_586);  add_343 = view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_433: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_346, primals_25);  primals_25 = None
        mul_434: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_433, 4096)
        sum_188: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [2], True)
        mul_435: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_433, mul_10);  mul_433 = None
        sum_189: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2], True);  mul_435 = None
        mul_436: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_10, sum_189);  sum_189 = None
        sub_123: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_434, sum_188);  mul_434 = sum_188 = None
        sub_124: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_123, mul_436);  sub_123 = mul_436 = None
        mul_437: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_37, sub_124);  div_37 = sub_124 = None
        mul_438: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_346, mul_10);  mul_10 = None
        sum_190: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 1]);  mul_438 = None
        sum_191: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 1]);  add_346 = None
        add_347: "f32[4096]" = torch.ops.aten.add.Tensor(add_325, sum_190);  add_325 = sum_190 = None
        add_348: "f32[4096]" = torch.ops.aten.add.Tensor(add_326, sum_191);  add_326 = sum_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_587: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_437, [4096, 4096])
        mm_136: "f32[4096, 16384]" = torch.ops.aten.mm.default(view_587, permute_143);  permute_143 = None
        permute_507: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_137: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
        sum_192: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        view_588: "f32[4096]" = torch.ops.aten.reshape.default(sum_192, [4096]);  sum_192 = None
        add_349: "f32[4096]" = torch.ops.aten.add.Tensor(add_327, view_588);  add_327 = view_588 = None
        add_350: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_328, mm_137);  add_328 = mm_137 = None
        view_589: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_136, [8, 512, 16384]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_21: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_5, [8, 512, 16384]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_439: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_589, mul_6);  mul_6 = None
        pow_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
        mul_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_21, mul_7);  mul_7 = None
        mul_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_11: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_440: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_589, add_11);  view_589 = add_11 = None
        mul_441: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_125: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_441);  mul_441 = None
        mul_442: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_439, sub_125);  mul_439 = sub_125 = None
        mul_443: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_442, 0.7978845608028654);  mul_442 = None
        mul_444: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_443, 0.044715)
        pow_26: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
        mul_445: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_446: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
        add_351: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_443, mul_446);  mul_443 = mul_446 = None
        mul_447: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_440, 0.5);  mul_440 = None
        add_352: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_351, mul_447);  add_351 = mul_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_590: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_352, [4096, 16384]);  add_352 = None
        mm_138: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_590, permute_147);  permute_147 = None
        permute_511: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_139: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
        sum_193: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f32[16384]" = torch.ops.aten.reshape.default(sum_193, [16384]);  sum_193 = None
        add_353: "f32[16384]" = torch.ops.aten.add.Tensor(add_331, view_591);  add_331 = view_591 = None
        add_354: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_332, mm_139);  add_332 = mm_139 = None
        view_592: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_138, [8, 512, 4096]);  mm_138 = None
        add_355: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_437, view_592);  mul_437 = view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_449: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_355, primals_19);  primals_19 = None
        mul_450: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_449, 4096)
        sum_194: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True)
        mul_451: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_449, mul_4);  mul_449 = None
        sum_195: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        mul_452: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_4, sum_195);  sum_195 = None
        sub_127: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_450, sum_194);  mul_450 = sum_194 = None
        sub_128: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_127, mul_452);  sub_127 = mul_452 = None
        mul_453: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_38, sub_128);  div_38 = sub_128 = None
        mul_454: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_355, mul_4);  mul_4 = None
        sum_196: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1]);  mul_454 = None
        sum_197: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_355, [0, 1]);  add_355 = None
        add_356: "f32[4096]" = torch.ops.aten.add.Tensor(add_334, sum_196);  add_334 = sum_196 = None
        add_357: "f32[4096]" = torch.ops.aten.add.Tensor(add_335, sum_197);  add_335 = sum_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_593: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_453, [4096, 4096])
        mm_140: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_593, permute_151);  permute_151 = None
        permute_515: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_141: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
        sum_198: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        view_594: "f32[4096]" = torch.ops.aten.reshape.default(sum_198, [4096]);  sum_198 = None
        add_358: "f32[4096]" = torch.ops.aten.add.Tensor(add_336, view_594);  add_336 = view_594 = None
        add_359: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_337, mm_141);  add_337 = mm_141 = None
        view_595: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_140, [8, 512, 4096]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_596: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(view_595, [8, 512, 64, 64]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_518: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_105: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_597: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_105, [512, 512, 64]);  clone_105 = None
        expand_5: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_1, [8, 64, 512, 512])
        view_14: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [512, 512, 512]);  expand_5 = None
        permute_519: "f32[512, 512, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        bmm_68: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_597);  permute_519 = None
        bmm_69: "f32[512, 512, 512]" = torch.ops.aten.bmm.default(view_597, permute_520);  view_597 = permute_520 = None
        view_598: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [8, 64, 512, 64]);  bmm_68 = None
        view_599: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, [8, 64, 512, 512]);  bmm_69 = None
        mul_455: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_599, where_1);  view_599 = None
        sum_199: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [-1], True)
        neg_12: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_11: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_12, sum_199, mul_455);  neg_12 = sum_199 = mul_455 = None
        view_600: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_11, [512, 512, 512]);  fma_11 = None
        bmm_70: "f32[512, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_600);  permute_521 = None
        bmm_71: "f32[512, 512, 64]" = torch.ops.aten.bmm.default(view_600, permute_522);  view_600 = permute_522 = None
        view_601: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, [8, 64, 64, 512]);  bmm_70 = None
        view_602: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [8, 64, 512, 64]);  bmm_71 = None
        mul_456: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_601, 0.3535533905932738);  view_601 = None
        permute_523: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_456, [0, 1, 3, 2]);  mul_456 = None
        mul_457: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_602, 0.3535533905932738);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_524: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        clone_106: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        view_603: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_106, [8, 512, 4096]);  clone_106 = None
        view_604: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_603, [4096, 4096]);  view_603 = None
        mm_142: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_604, permute_162);  permute_162 = None
        permute_526: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_604, [1, 0])
        mm_143: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_526, view_2);  permute_526 = None
        sum_200: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
        view_605: "f32[4096]" = torch.ops.aten.reshape.default(sum_200, [4096]);  sum_200 = None
        add_360: "f32[4096]" = torch.ops.aten.add.Tensor(add_338, view_605);  add_338 = view_605 = None
        add_361: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_339, mm_143);  add_339 = mm_143 = None
        view_606: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_142, [8, 512, 4096]);  mm_142 = None
        add_362: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_453, view_606);  mul_453 = view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_529: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
        view_607: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_529, [8, 512, 4096]);  permute_529 = None
        clone_107: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_607, memory_format = torch.contiguous_format);  view_607 = None
        view_608: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_107, [4096, 4096]);  clone_107 = None
        mm_144: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_608, permute_167);  permute_167 = None
        permute_531: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_145: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = None
        sum_201: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
        view_609: "f32[4096]" = torch.ops.aten.reshape.default(sum_201, [4096]);  sum_201 = None
        add_363: "f32[4096]" = torch.ops.aten.add.Tensor(add_341, view_609);  add_341 = view_609 = None
        add_364: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_342, mm_145);  add_342 = mm_145 = None
        view_610: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_144, [8, 512, 4096]);  mm_144 = None
        add_365: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_362, view_610);  add_362 = view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_534: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_457, [0, 2, 1, 3]);  mul_457 = None
        clone_108: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
        view_611: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_108, [8, 512, 4096]);  clone_108 = None
        view_612: "f32[4096, 4096]" = torch.ops.aten.reshape.default(view_611, [4096, 4096]);  view_611 = None
        mm_146: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_612, permute_172);  permute_172 = None
        permute_536: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_147: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
        sum_202: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        view_613: "f32[4096]" = torch.ops.aten.reshape.default(sum_202, [4096]);  sum_202 = None
        add_366: "f32[4096]" = torch.ops.aten.add.Tensor(add_344, view_613);  add_344 = view_613 = None
        add_367: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_345, mm_147);  add_345 = mm_147 = None
        view_614: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_146, [8, 512, 4096]);  mm_146 = None
        add_368: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_365, view_614);  add_365 = view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        view_615: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_368, [4096, 4096]);  add_368 = None
        permute: "f32[128, 4096]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_539: "f32[4096, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_148: "f32[4096, 128]" = torch.ops.aten.mm.default(view_615, permute_539);  permute_539 = None
        permute_540: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_149: "f32[4096, 128]" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
        sum_203: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        view_616: "f32[4096]" = torch.ops.aten.reshape.default(sum_203, [4096]);  sum_203 = None
        view_617: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(mm_148, [8, 512, 128]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_459: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_617, primals_7);  primals_7 = None
        mul_460: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_459, 128)
        sum_204: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
        mul_461: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_459, mul);  mul_459 = None
        sum_205: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
        mul_462: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, sum_205);  sum_205 = None
        sub_130: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_460, sum_204);  mul_460 = sum_204 = None
        sub_131: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_130, mul_462);  sub_130 = mul_462 = None
        mul_463: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_131);  div_39 = sub_131 = None
        mul_464: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_617, mul);  mul = None
        sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
        sum_207: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_617, [0, 1]);  view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:106 in forward, code: embeddings = embeddings + position_embeddings
        sum_208: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_463, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:105 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_12: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_12, -1);  eq_12 = None
        where_28: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, sum_208);  unsqueeze_5 = sum_208 = None
        full_default_42: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default_42, [primals_2], where_28, True);  full_default_42 = primals_2 = where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:96 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[8, 512]" = torch.ops.aten.expand.default(gather, [8, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:102 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_13: "b8[8, 512]" = torch.ops.aten.eq.Scalar(expand_1, -1)
        unsqueeze_6: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_13, -1);  eq_13 = None
        where_29: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_6, full_default_1, mul_463);  unsqueeze_6 = None
        full_default_44: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_44, [expand_1], where_29, True);  full_default_44 = expand_1 = where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:101 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_14: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_7: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_14, -1);  eq_14 = None
        where_30: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_7, full_default_1, mul_463);  unsqueeze_7 = full_default_1 = mul_463 = None
        full_default_46: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[30000, 128]" = torch.ops.aten.index_put.default(full_default_46, [primals_1], where_30, True);  full_default_46 = primals_1 = where_30 = None
        add_369: "f32[30000, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_2);  mm_1 = index_put_2 = None
        return (None, None, None, add_369, index_put_1, index_put, sum_206, sum_207, mm_149, view_616, add_367, add_366, add_364, add_363, add_361, add_360, add_359, add_358, add_356, add_357, add_354, add_353, add_350, add_349, add_347, add_348, mm_3, view_277, sum_20, sum_21, view_274, None)
