import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[64, 512]", primals_3: "i64[1, 512]", primals_5: "f32[30522, 128]", primals_8: "f32[128]", primals_10: "f32[256, 128]", primals_12: "f32[256, 256]", primals_14: "f32[256, 256]", primals_16: "f32[256, 256]", primals_18: "f32[256, 256]", primals_20: "f32[256]", primals_22: "f32[1024, 256]", primals_24: "f32[256, 1024]", primals_26: "f32[256]", primals_28: "f32[256, 256]", primals_30: "f32[256, 256]", primals_32: "f32[256, 256]", primals_34: "f32[256, 256]", primals_36: "f32[256]", primals_38: "f32[1024, 256]", primals_40: "f32[256, 1024]", primals_42: "f32[256]", primals_44: "f32[256, 256]", primals_46: "f32[256, 256]", primals_48: "f32[256, 256]", primals_50: "f32[256, 256]", primals_52: "f32[256]", primals_54: "f32[1024, 256]", primals_56: "f32[256, 1024]", primals_58: "f32[256]", primals_60: "f32[256, 256]", primals_62: "f32[256, 256]", primals_64: "f32[256, 256]", primals_66: "f32[256, 256]", primals_68: "f32[256]", primals_70: "f32[1024, 256]", primals_72: "f32[256, 1024]", primals_74: "f32[256]", primals_76: "f32[256, 256]", primals_78: "f32[256, 256]", primals_80: "f32[256, 256]", primals_82: "f32[256, 256]", primals_84: "f32[256]", primals_86: "f32[1024, 256]", primals_88: "f32[256, 1024]", primals_90: "f32[256]", primals_92: "f32[256, 256]", primals_94: "f32[256, 256]", primals_96: "f32[256, 256]", primals_98: "f32[256, 256]", primals_100: "f32[256]", primals_102: "f32[1024, 256]", primals_104: "f32[256, 1024]", primals_106: "f32[256]", primals_108: "f32[256, 256]", primals_110: "f32[256, 256]", primals_112: "f32[256, 256]", primals_114: "f32[256, 256]", primals_116: "f32[256]", primals_118: "f32[1024, 256]", primals_120: "f32[256, 1024]", primals_122: "f32[256]", primals_124: "f32[256, 256]", primals_126: "f32[256, 256]", primals_128: "f32[256, 256]", primals_130: "f32[256, 256]", primals_132: "f32[256]", primals_134: "f32[1024, 256]", primals_136: "f32[256, 1024]", primals_138: "f32[256]", primals_140: "f32[256, 256]", primals_142: "f32[256, 256]", primals_144: "f32[256, 256]", primals_146: "f32[256, 256]", primals_148: "f32[256]", primals_150: "f32[1024, 256]", primals_152: "f32[256, 1024]", primals_154: "f32[256]", primals_156: "f32[256, 256]", primals_158: "f32[256, 256]", primals_160: "f32[256, 256]", primals_162: "f32[256, 256]", primals_164: "f32[256]", primals_166: "f32[1024, 256]", primals_168: "f32[256, 1024]", primals_170: "f32[256]", primals_172: "f32[256, 256]", primals_174: "f32[256, 256]", primals_176: "f32[256, 256]", primals_178: "f32[256, 256]", primals_180: "f32[256]", primals_182: "f32[1024, 256]", primals_184: "f32[256, 1024]", primals_186: "f32[256]", primals_188: "f32[256, 256]", primals_190: "f32[256, 256]", primals_192: "f32[256, 256]", primals_194: "f32[256, 256]", primals_196: "f32[256]", primals_198: "f32[1024, 256]", primals_200: "f32[256, 1024]", primals_202: "f32[256]", primals_204: "f32[128, 256]", primals_206: "f32[128]", gather: "i64[1, 512]", mul: "f32[64, 512, 128]", gt: "b8[64, 512, 128]", view: "f32[32768, 128]", ge: "b8[1, 1, 512, 1]", view_2: "f32[32768, 256]", bmm: "f32[256, 512, 512]", amax: "f32[64, 4, 512, 1]", sum_1: "f32[64, 4, 512, 1]", logical_not_1: "b8[64, 4, 512, 1]", gt_1: "b8[64, 4, 512, 512]", view_18: "f32[32768, 256]", gt_2: "b8[64, 512, 256]", mul_10: "f32[64, 512, 256]", view_20: "f32[32768, 256]", addmm_5: "f32[32768, 1024]", view_22: "f32[32768, 1024]", gt_3: "b8[64, 512, 256]", mul_17: "f32[64, 512, 256]", view_24: "f32[32768, 256]", where_3: "f32[64, 4, 512, 512]", gt_4: "b8[64, 4, 512, 512]", view_40: "f32[32768, 256]", gt_5: "b8[64, 512, 256]", mul_25: "f32[64, 512, 256]", view_42: "f32[32768, 256]", addmm_11: "f32[32768, 1024]", view_44: "f32[32768, 1024]", gt_6: "b8[64, 512, 256]", mul_32: "f32[64, 512, 256]", view_46: "f32[32768, 256]", where_5: "f32[64, 4, 512, 512]", gt_7: "b8[64, 4, 512, 512]", view_62: "f32[32768, 256]", gt_8: "b8[64, 512, 256]", mul_40: "f32[64, 512, 256]", view_64: "f32[32768, 256]", addmm_17: "f32[32768, 1024]", view_66: "f32[32768, 1024]", gt_9: "b8[64, 512, 256]", mul_47: "f32[64, 512, 256]", view_68: "f32[32768, 256]", where_7: "f32[64, 4, 512, 512]", gt_10: "b8[64, 4, 512, 512]", view_84: "f32[32768, 256]", gt_11: "b8[64, 512, 256]", mul_55: "f32[64, 512, 256]", view_86: "f32[32768, 256]", addmm_23: "f32[32768, 1024]", view_88: "f32[32768, 1024]", gt_12: "b8[64, 512, 256]", mul_62: "f32[64, 512, 256]", view_90: "f32[32768, 256]", where_9: "f32[64, 4, 512, 512]", gt_13: "b8[64, 4, 512, 512]", view_106: "f32[32768, 256]", gt_14: "b8[64, 512, 256]", mul_70: "f32[64, 512, 256]", view_108: "f32[32768, 256]", addmm_29: "f32[32768, 1024]", view_110: "f32[32768, 1024]", gt_15: "b8[64, 512, 256]", mul_77: "f32[64, 512, 256]", view_112: "f32[32768, 256]", where_11: "f32[64, 4, 512, 512]", gt_16: "b8[64, 4, 512, 512]", view_128: "f32[32768, 256]", gt_17: "b8[64, 512, 256]", mul_85: "f32[64, 512, 256]", view_130: "f32[32768, 256]", addmm_35: "f32[32768, 1024]", view_132: "f32[32768, 1024]", gt_18: "b8[64, 512, 256]", mul_92: "f32[64, 512, 256]", view_134: "f32[32768, 256]", where_13: "f32[64, 4, 512, 512]", gt_19: "b8[64, 4, 512, 512]", view_150: "f32[32768, 256]", gt_20: "b8[64, 512, 256]", mul_100: "f32[64, 512, 256]", view_152: "f32[32768, 256]", addmm_41: "f32[32768, 1024]", view_154: "f32[32768, 1024]", gt_21: "b8[64, 512, 256]", mul_107: "f32[64, 512, 256]", view_156: "f32[32768, 256]", where_15: "f32[64, 4, 512, 512]", gt_22: "b8[64, 4, 512, 512]", view_172: "f32[32768, 256]", gt_23: "b8[64, 512, 256]", mul_115: "f32[64, 512, 256]", view_174: "f32[32768, 256]", addmm_47: "f32[32768, 1024]", view_176: "f32[32768, 1024]", gt_24: "b8[64, 512, 256]", mul_122: "f32[64, 512, 256]", view_178: "f32[32768, 256]", where_17: "f32[64, 4, 512, 512]", gt_25: "b8[64, 4, 512, 512]", view_194: "f32[32768, 256]", gt_26: "b8[64, 512, 256]", mul_130: "f32[64, 512, 256]", view_196: "f32[32768, 256]", addmm_53: "f32[32768, 1024]", view_198: "f32[32768, 1024]", gt_27: "b8[64, 512, 256]", mul_137: "f32[64, 512, 256]", view_200: "f32[32768, 256]", where_19: "f32[64, 4, 512, 512]", gt_28: "b8[64, 4, 512, 512]", view_216: "f32[32768, 256]", gt_29: "b8[64, 512, 256]", mul_145: "f32[64, 512, 256]", view_218: "f32[32768, 256]", addmm_59: "f32[32768, 1024]", view_220: "f32[32768, 1024]", gt_30: "b8[64, 512, 256]", mul_152: "f32[64, 512, 256]", view_222: "f32[32768, 256]", where_21: "f32[64, 4, 512, 512]", gt_31: "b8[64, 4, 512, 512]", view_238: "f32[32768, 256]", gt_32: "b8[64, 512, 256]", mul_160: "f32[64, 512, 256]", view_240: "f32[32768, 256]", addmm_65: "f32[32768, 1024]", view_242: "f32[32768, 1024]", gt_33: "b8[64, 512, 256]", mul_167: "f32[64, 512, 256]", view_244: "f32[32768, 256]", where_23: "f32[64, 4, 512, 512]", gt_34: "b8[64, 4, 512, 512]", view_260: "f32[32768, 256]", gt_35: "b8[64, 512, 256]", mul_175: "f32[64, 512, 256]", view_262: "f32[32768, 256]", addmm_71: "f32[32768, 1024]", view_264: "f32[32768, 1024]", gt_36: "b8[64, 512, 256]", mul_182: "f32[64, 512, 256]", view_266: "f32[32768, 256]", addmm_73: "f32[32768, 128]", getitem_51: "f32[64, 512, 1]", rsqrt_25: "f32[64, 512, 1]", view_268: "f32[32768, 128]", view_269: "f32[64, 512, 30522]", constant_pad_nd: "i64[64, 513]", amax_12: "f32[32768, 1]", log: "f32[32768, 1]", convert_element_type: "f32[]", div_15: "f32[64, 512, 1]", div_16: "f32[64, 512, 1]", permute_156: "f32[256, 512, 512]", permute_157: "f32[256, 64, 512]", permute_158: "f32[256, 64, 512]", permute_159: "f32[256, 512, 64]", div_17: "f32[64, 512, 1]", div_18: "f32[64, 512, 1]", permute_189: "f32[256, 512, 512]", permute_190: "f32[256, 64, 512]", permute_191: "f32[256, 64, 512]", permute_192: "f32[256, 512, 64]", div_19: "f32[64, 512, 1]", div_20: "f32[64, 512, 1]", permute_222: "f32[256, 512, 512]", permute_223: "f32[256, 64, 512]", permute_224: "f32[256, 64, 512]", permute_225: "f32[256, 512, 64]", div_21: "f32[64, 512, 1]", div_22: "f32[64, 512, 1]", permute_255: "f32[256, 512, 512]", permute_256: "f32[256, 64, 512]", permute_257: "f32[256, 64, 512]", permute_258: "f32[256, 512, 64]", div_23: "f32[64, 512, 1]", div_24: "f32[64, 512, 1]", permute_288: "f32[256, 512, 512]", permute_289: "f32[256, 64, 512]", permute_290: "f32[256, 64, 512]", permute_291: "f32[256, 512, 64]", div_25: "f32[64, 512, 1]", div_26: "f32[64, 512, 1]", permute_321: "f32[256, 512, 512]", permute_322: "f32[256, 64, 512]", permute_323: "f32[256, 64, 512]", permute_324: "f32[256, 512, 64]", div_27: "f32[64, 512, 1]", div_28: "f32[64, 512, 1]", permute_354: "f32[256, 512, 512]", permute_355: "f32[256, 64, 512]", permute_356: "f32[256, 64, 512]", permute_357: "f32[256, 512, 64]", div_29: "f32[64, 512, 1]", div_30: "f32[64, 512, 1]", permute_387: "f32[256, 512, 512]", permute_388: "f32[256, 64, 512]", permute_389: "f32[256, 64, 512]", permute_390: "f32[256, 512, 64]", div_31: "f32[64, 512, 1]", div_32: "f32[64, 512, 1]", permute_420: "f32[256, 512, 512]", permute_421: "f32[256, 64, 512]", permute_422: "f32[256, 64, 512]", permute_423: "f32[256, 512, 64]", div_33: "f32[64, 512, 1]", div_34: "f32[64, 512, 1]", permute_453: "f32[256, 512, 512]", permute_454: "f32[256, 64, 512]", permute_455: "f32[256, 64, 512]", permute_456: "f32[256, 512, 64]", div_35: "f32[64, 512, 1]", div_36: "f32[64, 512, 1]", permute_486: "f32[256, 512, 512]", permute_487: "f32[256, 64, 512]", permute_488: "f32[256, 64, 512]", permute_489: "f32[256, 512, 64]", div_37: "f32[64, 512, 1]", div_38: "f32[64, 512, 1]", permute_519: "f32[256, 512, 512]", permute_520: "f32[256, 64, 512]", permute_521: "f32[256, 64, 512]", permute_522: "f32[256, 512, 64]", div_39: "f32[64, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[64, 512, 30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[64, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_48: "i64[64, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_271: "i64[32768]" = torch.ops.aten.reshape.default(clone_48, [-1]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_4: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(view_271, 1);  view_271 = None
        ne_3: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[32768, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_36);  unsqueeze_4 = full_default_36 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[32768, 30522]" = torch.ops.aten.expand.default(where_26, [32768, 30522]);  where_26 = None
        eq_tensor: "b8[32768, 30522]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[32768, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_27: "f32[32768, 1]" = torch.ops.aten.where.self(ne_3, div_13, full_default_1);  ne_3 = div_13 = None
        mul_189: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_270: "f32[32768, 30522]" = torch.ops.aten.reshape.default(view_269, [-1, 30522]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_38: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = amax_12 = None
        sub_39: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        exp_13: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_16: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [1], True)
        mul_190: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_40: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_272: "f32[64, 512, 30522]" = torch.ops.aten.reshape.default(sub_40, [64, 512, 30522]);  sub_40 = None
        add_105: "f32[64, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_272);  tangents_2 = view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        view_273: "f32[32768, 30522]" = torch.ops.aten.reshape.default(add_105, [32768, 30522]);  add_105 = None
        permute_134: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_135: "f32[30522, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        constant_pad_nd_default: "f32[32768, 30524]" = torch.ops.aten.constant_pad_nd.default(view_273, [0, 2, 0, 0])
        constant_pad_nd_default_1: "f32[30524, 128]" = torch.ops.aten.constant_pad_nd.default(permute_135, [0, 0, 0, 2]);  permute_135 = None
        mm_default: "f32[32768, 128]" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_136: "f32[30522, 32768]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_1: "f32[30522, 128]" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
        sum_17: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        view_274: "f32[30522]" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
        view_275: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(mm_default, [64, 512, 128]);  mm_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_192: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, primals_206);  primals_206 = None
        mul_193: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_192, 128)
        sum_18: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_267: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [64, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_184: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_185: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476)
        erf_12: "f32[64, 512, 128]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_37: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_186, getitem_51);  mul_186 = getitem_51 = None
        mul_187: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_194: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_192, mul_187);  mul_192 = None
        sum_19: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_187, sum_19);  sum_19 = None
        sub_42: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_193, sum_18);  mul_193 = sum_18 = None
        sub_43: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub_42, mul_195);  sub_42 = mul_195 = None
        div_14: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_196: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(div_14, sub_43);  div_14 = sub_43 = None
        mul_197: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_187);  mul_187 = None
        sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_199: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
        mul_200: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, view_267)
        mul_201: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
        exp_14: "f32[64, 512, 128]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
        mul_202: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_203: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, mul_202);  view_267 = mul_202 = None
        add_107: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
        mul_204: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_196, add_107);  mul_196 = add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_276: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_204, [32768, 128]);  mul_204 = None
        permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        permute_139: "f32[128, 256]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_2: "f32[32768, 256]" = torch.ops.aten.mm.default(view_276, permute_139);  permute_139 = None
        permute_140: "f32[128, 32768]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_3: "f32[128, 256]" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
        sum_22: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        view_277: "f32[128]" = torch.ops.aten.reshape.default(sum_22, [128]);  sum_22 = None
        view_278: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_2, [64, 512, 256]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_206: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, primals_202);  primals_202 = None
        mul_207: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_206, 256)
        sum_23: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_206, mul_182);  mul_206 = None
        sum_24: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_182, sum_24);  sum_24 = None
        sub_45: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_207, sum_23);  mul_207 = sum_23 = None
        sub_46: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_45, mul_209);  sub_45 = mul_209 = None
        mul_210: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
        mul_211: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, mul_182);  mul_182 = None
        sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_212: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_213: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_210, mul_212);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_279: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_213, [32768, 256]);  mul_213 = None
        permute_132: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        permute_143: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_4: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_279, permute_143);  permute_143 = None
        permute_144: "f32[256, 32768]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_5: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
        sum_27: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        view_280: "f32[256]" = torch.ops.aten.reshape.default(sum_27, [256]);  sum_27 = None
        view_281: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_4, [64, 512, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_263: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_71, [64, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_178: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
        erf_11: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_215: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
        mul_216: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, view_263)
        mul_217: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_216, -0.5);  mul_216 = None
        exp_15: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_217);  mul_217 = None
        mul_218: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_219: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, mul_218);  view_263 = mul_218 = None
        add_109: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_215, mul_219);  mul_215 = mul_219 = None
        mul_220: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_281, add_109);  view_281 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_220, [32768, 1024]);  mul_220 = None
        permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_147: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_6: "f32[32768, 256]" = torch.ops.aten.mm.default(view_282, permute_147);  permute_147 = None
        permute_148: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_7: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
        sum_28: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283: "f32[1024]" = torch.ops.aten.reshape.default(sum_28, [1024]);  sum_28 = None
        view_284: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_6, [64, 512, 256]);  mm_6 = None
        add_110: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_210, view_284);  mul_210 = view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_222: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_110, primals_196);  primals_196 = None
        mul_223: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_222, 256)
        sum_29: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_222, mul_175);  mul_222 = None
        sum_30: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_175, sum_30);  sum_30 = None
        sub_48: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_223, sum_29);  mul_223 = sum_29 = None
        sub_49: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_48, mul_225);  sub_48 = mul_225 = None
        mul_226: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
        mul_227: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_110, mul_175);  mul_175 = None
        sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_228: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_229: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_226, mul_228);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_285: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_229, [32768, 256]);  mul_229 = None
        permute_130: "f32[256, 256]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        permute_151: "f32[256, 256]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_8: "f32[32768, 256]" = torch.ops.aten.mm.default(view_285, permute_151);  permute_151 = None
        permute_152: "f32[256, 32768]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_9: "f32[256, 256]" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
        sum_33: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_286: "f32[256]" = torch.ops.aten.reshape.default(sum_33, [256]);  sum_33 = None
        view_287: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_8, [64, 512, 256]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_288: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_287, [64, 512, 4, 64]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_51: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_289: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_51, [256, 512, 64]);  clone_51 = None
        bmm_24: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_156, view_289);  permute_156 = None
        bmm_25: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_289, permute_157);  view_289 = permute_157 = None
        view_290: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_24, [64, 4, 512, 64]);  bmm_24 = None
        view_291: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_25, [64, 4, 512, 512]);  bmm_25 = None
        convert_element_type_3: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_230: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_231: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_291, mul_230);  view_291 = mul_230 = None
        mul_232: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_231, where_23);  mul_231 = None
        sum_34: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [-1], True)
        neg_1: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_23);  where_23 = None
        fma: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_232);  neg_1 = sum_34 = mul_232 = None
        view_292: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma, [256, 512, 512]);  fma = None
        bmm_26: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_158, view_292);  permute_158 = None
        bmm_27: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_292, permute_159);  view_292 = permute_159 = None
        view_293: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_26, [64, 4, 64, 512]);  bmm_26 = None
        view_294: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [64, 4, 512, 64]);  bmm_27 = None
        mul_233: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_293, 0.3535533905932738);  view_293 = None
        permute_160: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_233, [0, 1, 3, 2]);  mul_233 = None
        mul_234: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_294, 0.3535533905932738);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_161: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_53: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_295: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_53, [64, 512, 256]);  clone_53 = None
        view_296: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_295, [32768, 256]);  view_295 = None
        permute_126: "f32[256, 256]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_162: "f32[256, 256]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        mm_10: "f32[32768, 256]" = torch.ops.aten.mm.default(view_296, permute_162);  permute_162 = None
        permute_163: "f32[256, 32768]" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_11: "f32[256, 256]" = torch.ops.aten.mm.default(permute_163, view_244);  permute_163 = None
        sum_35: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        view_297: "f32[256]" = torch.ops.aten.reshape.default(sum_35, [256]);  sum_35 = None
        view_298: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_10, [64, 512, 256]);  mm_10 = None
        add_111: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_226, view_298);  mul_226 = view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_166: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
        view_299: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_166, [64, 512, 256]);  permute_166 = None
        clone_54: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_299, memory_format = torch.contiguous_format);  view_299 = None
        view_300: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_54, [32768, 256]);  clone_54 = None
        permute_124: "f32[256, 256]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_167: "f32[256, 256]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        mm_12: "f32[32768, 256]" = torch.ops.aten.mm.default(view_300, permute_167);  permute_167 = None
        permute_168: "f32[256, 32768]" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_13: "f32[256, 256]" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = None
        sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        view_301: "f32[256]" = torch.ops.aten.reshape.default(sum_36, [256]);  sum_36 = None
        view_302: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_12, [64, 512, 256]);  mm_12 = None
        add_112: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_111, view_302);  add_111 = view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_171: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_234, [0, 2, 1, 3]);  mul_234 = None
        clone_55: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_303: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_55, [64, 512, 256]);  clone_55 = None
        view_304: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_303, [32768, 256]);  view_303 = None
        permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_172: "f32[256, 256]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_14: "f32[32768, 256]" = torch.ops.aten.mm.default(view_304, permute_172);  permute_172 = None
        permute_173: "f32[256, 32768]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_15: "f32[256, 256]" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
        sum_37: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        view_305: "f32[256]" = torch.ops.aten.reshape.default(sum_37, [256]);  sum_37 = None
        view_306: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_14, [64, 512, 256]);  mm_14 = None
        add_113: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_112, view_306);  add_112 = view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_236: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_113, primals_186);  primals_186 = None
        mul_237: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_236, 256)
        sum_38: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_236, mul_167);  mul_236 = None
        sum_39: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_167, sum_39);  sum_39 = None
        sub_51: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_237, sum_38);  mul_237 = sum_38 = None
        sub_52: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_51, mul_239);  sub_51 = mul_239 = None
        mul_240: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_17, sub_52);  div_17 = sub_52 = None
        mul_241: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_113, mul_167);  mul_167 = None
        sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_242: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_243: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_240, mul_242);  mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_307: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_243, [32768, 256]);  mul_243 = None
        permute_121: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_176: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_16: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_307, permute_176);  permute_176 = None
        permute_177: "f32[256, 32768]" = torch.ops.aten.permute.default(view_307, [1, 0])
        mm_17: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
        sum_42: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        view_308: "f32[256]" = torch.ops.aten.reshape.default(sum_42, [256]);  sum_42 = None
        view_309: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_16, [64, 512, 1024]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_241: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_65, [64, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_163: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476)
        erf_10: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_245: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_246: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, view_241)
        mul_247: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
        exp_16: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
        mul_248: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_249: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, mul_248);  view_241 = mul_248 = None
        add_115: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
        mul_250: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_309, add_115);  view_309 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_310: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_250, [32768, 1024]);  mul_250 = None
        permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_180: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_18: "f32[32768, 256]" = torch.ops.aten.mm.default(view_310, permute_180);  permute_180 = None
        permute_181: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_19: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
        sum_43: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        view_311: "f32[1024]" = torch.ops.aten.reshape.default(sum_43, [1024]);  sum_43 = None
        view_312: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_18, [64, 512, 256]);  mm_18 = None
        add_116: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_240, view_312);  mul_240 = view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_252: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_116, primals_180);  primals_180 = None
        mul_253: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_252, 256)
        sum_44: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_252, mul_160);  mul_252 = None
        sum_45: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_160, sum_45);  sum_45 = None
        sub_54: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_253, sum_44);  mul_253 = sum_44 = None
        sub_55: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_54, mul_255);  sub_54 = mul_255 = None
        mul_256: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_18, sub_55);  div_18 = sub_55 = None
        mul_257: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_116, mul_160);  mul_160 = None
        sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_5: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_258: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_259: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_256, mul_258);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_313: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_259, [32768, 256]);  mul_259 = None
        permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_184: "f32[256, 256]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_20: "f32[32768, 256]" = torch.ops.aten.mm.default(view_313, permute_184);  permute_184 = None
        permute_185: "f32[256, 32768]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_21: "f32[256, 256]" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
        sum_48: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
        view_314: "f32[256]" = torch.ops.aten.reshape.default(sum_48, [256]);  sum_48 = None
        view_315: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_20, [64, 512, 256]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_316: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_315, [64, 512, 4, 64]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_188: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_58: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_317: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_58, [256, 512, 64]);  clone_58 = None
        bmm_28: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_189, view_317);  permute_189 = None
        bmm_29: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_317, permute_190);  view_317 = permute_190 = None
        view_318: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_28, [64, 4, 512, 64]);  bmm_28 = None
        view_319: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_29, [64, 4, 512, 512]);  bmm_29 = None
        convert_element_type_6: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_260: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_261: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_319, mul_260);  view_319 = mul_260 = None
        mul_262: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_261, where_21);  mul_261 = None
        sum_49: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_2: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_21);  where_21 = None
        fma_1: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_262);  neg_2 = sum_49 = mul_262 = None
        view_320: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_1, [256, 512, 512]);  fma_1 = None
        bmm_30: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_191, view_320);  permute_191 = None
        bmm_31: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_320, permute_192);  view_320 = permute_192 = None
        view_321: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_30, [64, 4, 64, 512]);  bmm_30 = None
        view_322: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [64, 4, 512, 64]);  bmm_31 = None
        mul_263: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_321, 0.3535533905932738);  view_321 = None
        permute_193: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_263, [0, 1, 3, 2]);  mul_263 = None
        mul_264: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_322, 0.3535533905932738);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_194: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_60: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_323: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_60, [64, 512, 256]);  clone_60 = None
        view_324: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_323, [32768, 256]);  view_323 = None
        permute_115: "f32[256, 256]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_195: "f32[256, 256]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        mm_22: "f32[32768, 256]" = torch.ops.aten.mm.default(view_324, permute_195);  permute_195 = None
        permute_196: "f32[256, 32768]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_196, view_222);  permute_196 = None
        sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325: "f32[256]" = torch.ops.aten.reshape.default(sum_50, [256]);  sum_50 = None
        view_326: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_22, [64, 512, 256]);  mm_22 = None
        add_117: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_256, view_326);  mul_256 = view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_199: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
        view_327: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_199, [64, 512, 256]);  permute_199 = None
        clone_61: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_327, memory_format = torch.contiguous_format);  view_327 = None
        view_328: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_61, [32768, 256]);  clone_61 = None
        permute_113: "f32[256, 256]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_200: "f32[256, 256]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        mm_24: "f32[32768, 256]" = torch.ops.aten.mm.default(view_328, permute_200);  permute_200 = None
        permute_201: "f32[256, 32768]" = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = None
        sum_51: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        view_329: "f32[256]" = torch.ops.aten.reshape.default(sum_51, [256]);  sum_51 = None
        view_330: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_24, [64, 512, 256]);  mm_24 = None
        add_118: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_117, view_330);  add_117 = view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_204: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_264, [0, 2, 1, 3]);  mul_264 = None
        clone_62: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_331: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_62, [64, 512, 256]);  clone_62 = None
        view_332: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_331, [32768, 256]);  view_331 = None
        permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_205: "f32[256, 256]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_26: "f32[32768, 256]" = torch.ops.aten.mm.default(view_332, permute_205);  permute_205 = None
        permute_206: "f32[256, 32768]" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_27: "f32[256, 256]" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
        sum_52: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
        view_333: "f32[256]" = torch.ops.aten.reshape.default(sum_52, [256]);  sum_52 = None
        view_334: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_26, [64, 512, 256]);  mm_26 = None
        add_119: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_118, view_334);  add_118 = view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_266: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_119, primals_170);  primals_170 = None
        mul_267: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_266, 256)
        sum_53: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
        mul_268: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_266, mul_152);  mul_266 = None
        sum_54: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
        mul_269: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_152, sum_54);  sum_54 = None
        sub_57: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_267, sum_53);  mul_267 = sum_53 = None
        sub_58: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_57, mul_269);  sub_57 = mul_269 = None
        mul_270: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_19, sub_58);  div_19 = sub_58 = None
        mul_271: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_119, mul_152);  mul_152 = None
        sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
        sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_7: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_272: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_273: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, mul_272);  mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_335: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_273, [32768, 256]);  mul_273 = None
        permute_110: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        permute_209: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_28: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_335, permute_209);  permute_209 = None
        permute_210: "f32[256, 32768]" = torch.ops.aten.permute.default(view_335, [1, 0])
        mm_29: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
        sum_57: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        view_336: "f32[256]" = torch.ops.aten.reshape.default(sum_57, [256]);  sum_57 = None
        view_337: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_28, [64, 512, 1024]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_219: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_59, [64, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_148: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476)
        erf_9: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_275: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_276: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, view_219)
        mul_277: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
        exp_17: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
        mul_278: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_279: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, mul_278);  view_219 = mul_278 = None
        add_121: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
        mul_280: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_337, add_121);  view_337 = add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_338: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_280, [32768, 1024]);  mul_280 = None
        permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        permute_213: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_30: "f32[32768, 256]" = torch.ops.aten.mm.default(view_338, permute_213);  permute_213 = None
        permute_214: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_31: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
        sum_58: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        view_339: "f32[1024]" = torch.ops.aten.reshape.default(sum_58, [1024]);  sum_58 = None
        view_340: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_30, [64, 512, 256]);  mm_30 = None
        add_122: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_270, view_340);  mul_270 = view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_282: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_122, primals_164);  primals_164 = None
        mul_283: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, 256)
        sum_59: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
        mul_284: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, mul_145);  mul_282 = None
        sum_60: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        mul_285: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_145, sum_60);  sum_60 = None
        sub_60: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_283, sum_59);  mul_283 = sum_59 = None
        sub_61: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_60, mul_285);  sub_60 = mul_285 = None
        mul_286: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_20, sub_61);  div_20 = sub_61 = None
        mul_287: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_122, mul_145);  mul_145 = None
        sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_288: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_289: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_341: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_289, [32768, 256]);  mul_289 = None
        permute_108: "f32[256, 256]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_217: "f32[256, 256]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_32: "f32[32768, 256]" = torch.ops.aten.mm.default(view_341, permute_217);  permute_217 = None
        permute_218: "f32[256, 32768]" = torch.ops.aten.permute.default(view_341, [1, 0])
        mm_33: "f32[256, 256]" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
        sum_63: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        view_342: "f32[256]" = torch.ops.aten.reshape.default(sum_63, [256]);  sum_63 = None
        view_343: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_32, [64, 512, 256]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_344: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_343, [64, 512, 4, 64]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_65: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_345: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_65, [256, 512, 64]);  clone_65 = None
        bmm_32: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_222, view_345);  permute_222 = None
        bmm_33: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_345, permute_223);  view_345 = permute_223 = None
        view_346: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_32, [64, 4, 512, 64]);  bmm_32 = None
        view_347: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_33, [64, 4, 512, 512]);  bmm_33 = None
        convert_element_type_9: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_290: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_291: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_347, mul_290);  view_347 = mul_290 = None
        mul_292: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_291, where_19);  mul_291 = None
        sum_64: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
        neg_3: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_19);  where_19 = None
        fma_2: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_292);  neg_3 = sum_64 = mul_292 = None
        view_348: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_2, [256, 512, 512]);  fma_2 = None
        bmm_34: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_224, view_348);  permute_224 = None
        bmm_35: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_348, permute_225);  view_348 = permute_225 = None
        view_349: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_34, [64, 4, 64, 512]);  bmm_34 = None
        view_350: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [64, 4, 512, 64]);  bmm_35 = None
        mul_293: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_349, 0.3535533905932738);  view_349 = None
        permute_226: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_293, [0, 1, 3, 2]);  mul_293 = None
        mul_294: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_350, 0.3535533905932738);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_227: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_67: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_351: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_67, [64, 512, 256]);  clone_67 = None
        view_352: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_351, [32768, 256]);  view_351 = None
        permute_104: "f32[256, 256]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_228: "f32[256, 256]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        mm_34: "f32[32768, 256]" = torch.ops.aten.mm.default(view_352, permute_228);  permute_228 = None
        permute_229: "f32[256, 32768]" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_35: "f32[256, 256]" = torch.ops.aten.mm.default(permute_229, view_200);  permute_229 = None
        sum_65: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        view_353: "f32[256]" = torch.ops.aten.reshape.default(sum_65, [256]);  sum_65 = None
        view_354: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_34, [64, 512, 256]);  mm_34 = None
        add_123: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_286, view_354);  mul_286 = view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_232: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
        view_355: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_232, [64, 512, 256]);  permute_232 = None
        clone_68: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_355, memory_format = torch.contiguous_format);  view_355 = None
        view_356: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_68, [32768, 256]);  clone_68 = None
        permute_102: "f32[256, 256]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        permute_233: "f32[256, 256]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        mm_36: "f32[32768, 256]" = torch.ops.aten.mm.default(view_356, permute_233);  permute_233 = None
        permute_234: "f32[256, 32768]" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_37: "f32[256, 256]" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = None
        sum_66: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        view_357: "f32[256]" = torch.ops.aten.reshape.default(sum_66, [256]);  sum_66 = None
        view_358: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_36, [64, 512, 256]);  mm_36 = None
        add_124: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_123, view_358);  add_123 = view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_237: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_294, [0, 2, 1, 3]);  mul_294 = None
        clone_69: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_359: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_69, [64, 512, 256]);  clone_69 = None
        view_360: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_359, [32768, 256]);  view_359 = None
        permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_238: "f32[256, 256]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_38: "f32[32768, 256]" = torch.ops.aten.mm.default(view_360, permute_238);  permute_238 = None
        permute_239: "f32[256, 32768]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_39: "f32[256, 256]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
        sum_67: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        view_361: "f32[256]" = torch.ops.aten.reshape.default(sum_67, [256]);  sum_67 = None
        view_362: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_38, [64, 512, 256]);  mm_38 = None
        add_125: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_124, view_362);  add_124 = view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_296: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_125, primals_154);  primals_154 = None
        mul_297: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_296, 256)
        sum_68: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
        mul_298: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_296, mul_137);  mul_296 = None
        sum_69: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, sum_69);  sum_69 = None
        sub_63: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_297, sum_68);  mul_297 = sum_68 = None
        sub_64: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_63, mul_299);  sub_63 = mul_299 = None
        mul_300: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_21, sub_64);  div_21 = sub_64 = None
        mul_301: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_125, mul_137);  mul_137 = None
        sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
        sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_302: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_303: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_300, mul_302);  mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_363: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_303, [32768, 256]);  mul_303 = None
        permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_242: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_40: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_363, permute_242);  permute_242 = None
        permute_243: "f32[256, 32768]" = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_41: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
        sum_72: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        view_364: "f32[256]" = torch.ops.aten.reshape.default(sum_72, [256]);  sum_72 = None
        view_365: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_40, [64, 512, 1024]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_197: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_53, [64, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_133: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
        erf_8: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_305: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_306: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, view_197)
        mul_307: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
        exp_18: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
        mul_308: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_309: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, mul_308);  view_197 = mul_308 = None
        add_127: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
        mul_310: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_365, add_127);  view_365 = add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_366: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_310, [32768, 1024]);  mul_310 = None
        permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_246: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_42: "f32[32768, 256]" = torch.ops.aten.mm.default(view_366, permute_246);  permute_246 = None
        permute_247: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_43: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
        sum_73: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        view_367: "f32[1024]" = torch.ops.aten.reshape.default(sum_73, [1024]);  sum_73 = None
        view_368: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_42, [64, 512, 256]);  mm_42 = None
        add_128: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_300, view_368);  mul_300 = view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_312: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_128, primals_148);  primals_148 = None
        mul_313: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, 256)
        sum_74: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
        mul_314: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, mul_130);  mul_312 = None
        sum_75: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
        mul_315: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_130, sum_75);  sum_75 = None
        sub_66: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_313, sum_74);  mul_313 = sum_74 = None
        sub_67: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_66, mul_315);  sub_66 = mul_315 = None
        mul_316: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_22, sub_67);  div_22 = sub_67 = None
        mul_317: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_128, mul_130);  mul_130 = None
        sum_76: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
        sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_11: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_318: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_319: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_316, mul_318);  mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_369: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_319, [32768, 256]);  mul_319 = None
        permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_250: "f32[256, 256]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_44: "f32[32768, 256]" = torch.ops.aten.mm.default(view_369, permute_250);  permute_250 = None
        permute_251: "f32[256, 32768]" = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_45: "f32[256, 256]" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
        sum_78: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        view_370: "f32[256]" = torch.ops.aten.reshape.default(sum_78, [256]);  sum_78 = None
        view_371: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_44, [64, 512, 256]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_372: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_371, [64, 512, 4, 64]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_254: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_72: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_373: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_72, [256, 512, 64]);  clone_72 = None
        bmm_36: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_255, view_373);  permute_255 = None
        bmm_37: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_373, permute_256);  view_373 = permute_256 = None
        view_374: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_36, [64, 4, 512, 64]);  bmm_36 = None
        view_375: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_37, [64, 4, 512, 512]);  bmm_37 = None
        convert_element_type_12: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_320: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_321: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_375, mul_320);  view_375 = mul_320 = None
        mul_322: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_321, where_17);  mul_321 = None
        sum_79: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
        neg_4: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_17);  where_17 = None
        fma_3: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_322);  neg_4 = sum_79 = mul_322 = None
        view_376: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_3, [256, 512, 512]);  fma_3 = None
        bmm_38: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_257, view_376);  permute_257 = None
        bmm_39: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_376, permute_258);  view_376 = permute_258 = None
        view_377: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_38, [64, 4, 64, 512]);  bmm_38 = None
        view_378: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [64, 4, 512, 64]);  bmm_39 = None
        mul_323: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_377, 0.3535533905932738);  view_377 = None
        permute_259: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_323, [0, 1, 3, 2]);  mul_323 = None
        mul_324: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_378, 0.3535533905932738);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_260: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_74: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_379: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_74, [64, 512, 256]);  clone_74 = None
        view_380: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_379, [32768, 256]);  view_379 = None
        permute_93: "f32[256, 256]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_261: "f32[256, 256]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        mm_46: "f32[32768, 256]" = torch.ops.aten.mm.default(view_380, permute_261);  permute_261 = None
        permute_262: "f32[256, 32768]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_262, view_178);  permute_262 = None
        sum_80: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        view_381: "f32[256]" = torch.ops.aten.reshape.default(sum_80, [256]);  sum_80 = None
        view_382: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_46, [64, 512, 256]);  mm_46 = None
        add_129: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_316, view_382);  mul_316 = view_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_265: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
        view_383: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_265, [64, 512, 256]);  permute_265 = None
        clone_75: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_383, memory_format = torch.contiguous_format);  view_383 = None
        view_384: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_75, [32768, 256]);  clone_75 = None
        permute_91: "f32[256, 256]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_266: "f32[256, 256]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        mm_48: "f32[32768, 256]" = torch.ops.aten.mm.default(view_384, permute_266);  permute_266 = None
        permute_267: "f32[256, 32768]" = torch.ops.aten.permute.default(view_384, [1, 0])
        mm_49: "f32[256, 256]" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = None
        sum_81: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
        view_385: "f32[256]" = torch.ops.aten.reshape.default(sum_81, [256]);  sum_81 = None
        view_386: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_48, [64, 512, 256]);  mm_48 = None
        add_130: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_129, view_386);  add_129 = view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_270: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_324, [0, 2, 1, 3]);  mul_324 = None
        clone_76: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_387: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_76, [64, 512, 256]);  clone_76 = None
        view_388: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_387, [32768, 256]);  view_387 = None
        permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_271: "f32[256, 256]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_50: "f32[32768, 256]" = torch.ops.aten.mm.default(view_388, permute_271);  permute_271 = None
        permute_272: "f32[256, 32768]" = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_51: "f32[256, 256]" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
        sum_82: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        view_389: "f32[256]" = torch.ops.aten.reshape.default(sum_82, [256]);  sum_82 = None
        view_390: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_50, [64, 512, 256]);  mm_50 = None
        add_131: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_130, view_390);  add_130 = view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_326: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_131, primals_138);  primals_138 = None
        mul_327: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_326, 256)
        sum_83: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_326, mul_122);  mul_326 = None
        sum_84: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_122, sum_84);  sum_84 = None
        sub_69: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_327, sum_83);  mul_327 = sum_83 = None
        sub_70: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_69, mul_329);  sub_69 = mul_329 = None
        mul_330: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_23, sub_70);  div_23 = sub_70 = None
        mul_331: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_131, mul_122);  mul_122 = None
        sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_332: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_333: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_330, mul_332);  mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_391: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_333, [32768, 256]);  mul_333 = None
        permute_88: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_275: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_52: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_391, permute_275);  permute_275 = None
        permute_276: "f32[256, 32768]" = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_53: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
        sum_87: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        view_392: "f32[256]" = torch.ops.aten.reshape.default(sum_87, [256]);  sum_87 = None
        view_393: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_52, [64, 512, 1024]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_175: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_47, [64, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_118: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476)
        erf_7: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_335: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_336: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, view_175)
        mul_337: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_336, -0.5);  mul_336 = None
        exp_19: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_337);  mul_337 = None
        mul_338: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_339: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, mul_338);  view_175 = mul_338 = None
        add_133: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_335, mul_339);  mul_335 = mul_339 = None
        mul_340: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_393, add_133);  view_393 = add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_340, [32768, 1024]);  mul_340 = None
        permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_279: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_54: "f32[32768, 256]" = torch.ops.aten.mm.default(view_394, permute_279);  permute_279 = None
        permute_280: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_55: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
        sum_88: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        view_395: "f32[1024]" = torch.ops.aten.reshape.default(sum_88, [1024]);  sum_88 = None
        view_396: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_54, [64, 512, 256]);  mm_54 = None
        add_134: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_330, view_396);  mul_330 = view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_342: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_134, primals_132);  primals_132 = None
        mul_343: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_342, 256)
        sum_89: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_342, mul_115);  mul_342 = None
        sum_90: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_115, sum_90);  sum_90 = None
        sub_72: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_343, sum_89);  mul_343 = sum_89 = None
        sub_73: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_72, mul_345);  sub_72 = mul_345 = None
        mul_346: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_24, sub_73);  div_24 = sub_73 = None
        mul_347: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_134, mul_115);  mul_115 = None
        sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_348: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_349: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_346, mul_348);  mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_397: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_349, [32768, 256]);  mul_349 = None
        permute_86: "f32[256, 256]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_283: "f32[256, 256]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_56: "f32[32768, 256]" = torch.ops.aten.mm.default(view_397, permute_283);  permute_283 = None
        permute_284: "f32[256, 32768]" = torch.ops.aten.permute.default(view_397, [1, 0])
        mm_57: "f32[256, 256]" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
        sum_93: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        view_398: "f32[256]" = torch.ops.aten.reshape.default(sum_93, [256]);  sum_93 = None
        view_399: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_56, [64, 512, 256]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_400: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_399, [64, 512, 4, 64]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_79: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_401: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_79, [256, 512, 64]);  clone_79 = None
        bmm_40: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_288, view_401);  permute_288 = None
        bmm_41: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_401, permute_289);  view_401 = permute_289 = None
        view_402: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_40, [64, 4, 512, 64]);  bmm_40 = None
        view_403: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_41, [64, 4, 512, 512]);  bmm_41 = None
        convert_element_type_15: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_350: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_351: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_403, mul_350);  view_403 = mul_350 = None
        mul_352: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_351, where_15);  mul_351 = None
        sum_94: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_5: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_15);  where_15 = None
        fma_4: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_352);  neg_5 = sum_94 = mul_352 = None
        view_404: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_4, [256, 512, 512]);  fma_4 = None
        bmm_42: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_290, view_404);  permute_290 = None
        bmm_43: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_404, permute_291);  view_404 = permute_291 = None
        view_405: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_42, [64, 4, 64, 512]);  bmm_42 = None
        view_406: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [64, 4, 512, 64]);  bmm_43 = None
        mul_353: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_405, 0.3535533905932738);  view_405 = None
        permute_292: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_353, [0, 1, 3, 2]);  mul_353 = None
        mul_354: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_406, 0.3535533905932738);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_293: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_81: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_407: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_81, [64, 512, 256]);  clone_81 = None
        view_408: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_407, [32768, 256]);  view_407 = None
        permute_82: "f32[256, 256]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_294: "f32[256, 256]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        mm_58: "f32[32768, 256]" = torch.ops.aten.mm.default(view_408, permute_294);  permute_294 = None
        permute_295: "f32[256, 32768]" = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_59: "f32[256, 256]" = torch.ops.aten.mm.default(permute_295, view_156);  permute_295 = None
        sum_95: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        view_409: "f32[256]" = torch.ops.aten.reshape.default(sum_95, [256]);  sum_95 = None
        view_410: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_58, [64, 512, 256]);  mm_58 = None
        add_135: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_346, view_410);  mul_346 = view_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_298: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
        view_411: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_298, [64, 512, 256]);  permute_298 = None
        clone_82: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_411, memory_format = torch.contiguous_format);  view_411 = None
        view_412: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_82, [32768, 256]);  clone_82 = None
        permute_80: "f32[256, 256]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_299: "f32[256, 256]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        mm_60: "f32[32768, 256]" = torch.ops.aten.mm.default(view_412, permute_299);  permute_299 = None
        permute_300: "f32[256, 32768]" = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_61: "f32[256, 256]" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = None
        sum_96: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        view_413: "f32[256]" = torch.ops.aten.reshape.default(sum_96, [256]);  sum_96 = None
        view_414: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_60, [64, 512, 256]);  mm_60 = None
        add_136: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_135, view_414);  add_135 = view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_303: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_354, [0, 2, 1, 3]);  mul_354 = None
        clone_83: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_415: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_83, [64, 512, 256]);  clone_83 = None
        view_416: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_415, [32768, 256]);  view_415 = None
        permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_304: "f32[256, 256]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_62: "f32[32768, 256]" = torch.ops.aten.mm.default(view_416, permute_304);  permute_304 = None
        permute_305: "f32[256, 32768]" = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_63: "f32[256, 256]" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
        sum_97: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        view_417: "f32[256]" = torch.ops.aten.reshape.default(sum_97, [256]);  sum_97 = None
        view_418: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_62, [64, 512, 256]);  mm_62 = None
        add_137: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_136, view_418);  add_136 = view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_356: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_137, primals_122);  primals_122 = None
        mul_357: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_356, 256)
        sum_98: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_356, mul_107);  mul_356 = None
        sum_99: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_107, sum_99);  sum_99 = None
        sub_75: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_357, sum_98);  mul_357 = sum_98 = None
        sub_76: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_75, mul_359);  sub_75 = mul_359 = None
        mul_360: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_25, sub_76);  div_25 = sub_76 = None
        mul_361: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_137, mul_107);  mul_107 = None
        sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_101: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_362: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_363: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_360, mul_362);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_419: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_363, [32768, 256]);  mul_363 = None
        permute_77: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_308: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_64: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_419, permute_308);  permute_308 = None
        permute_309: "f32[256, 32768]" = torch.ops.aten.permute.default(view_419, [1, 0])
        mm_65: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
        sum_102: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        view_420: "f32[256]" = torch.ops.aten.reshape.default(sum_102, [256]);  sum_102 = None
        view_421: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_64, [64, 512, 1024]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_153: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_41, [64, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_103: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476)
        erf_6: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_365: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
        mul_366: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, view_153)
        mul_367: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_366, -0.5);  mul_366 = None
        exp_20: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_367);  mul_367 = None
        mul_368: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_369: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, mul_368);  view_153 = mul_368 = None
        add_139: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_365, mul_369);  mul_365 = mul_369 = None
        mul_370: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_421, add_139);  view_421 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_422: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_370, [32768, 1024]);  mul_370 = None
        permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_312: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_66: "f32[32768, 256]" = torch.ops.aten.mm.default(view_422, permute_312);  permute_312 = None
        permute_313: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_67: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
        sum_103: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        view_423: "f32[1024]" = torch.ops.aten.reshape.default(sum_103, [1024]);  sum_103 = None
        view_424: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_66, [64, 512, 256]);  mm_66 = None
        add_140: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_360, view_424);  mul_360 = view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_372: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_140, primals_116);  primals_116 = None
        mul_373: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_372, 256)
        sum_104: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
        mul_374: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_372, mul_100);  mul_372 = None
        sum_105: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
        mul_375: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_100, sum_105);  sum_105 = None
        sub_78: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_373, sum_104);  mul_373 = sum_104 = None
        sub_79: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_78, mul_375);  sub_78 = mul_375 = None
        mul_376: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_26, sub_79);  div_26 = sub_79 = None
        mul_377: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_140, mul_100);  mul_100 = None
        sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
        sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_378: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_379: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_376, mul_378);  mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_425: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_379, [32768, 256]);  mul_379 = None
        permute_75: "f32[256, 256]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_316: "f32[256, 256]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_68: "f32[32768, 256]" = torch.ops.aten.mm.default(view_425, permute_316);  permute_316 = None
        permute_317: "f32[256, 32768]" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_69: "f32[256, 256]" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
        sum_108: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        view_426: "f32[256]" = torch.ops.aten.reshape.default(sum_108, [256]);  sum_108 = None
        view_427: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_68, [64, 512, 256]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_428: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_427, [64, 512, 4, 64]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_320: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_86: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_429: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_86, [256, 512, 64]);  clone_86 = None
        bmm_44: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_321, view_429);  permute_321 = None
        bmm_45: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_429, permute_322);  view_429 = permute_322 = None
        view_430: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_44, [64, 4, 512, 64]);  bmm_44 = None
        view_431: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_45, [64, 4, 512, 512]);  bmm_45 = None
        convert_element_type_18: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_380: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_381: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_431, mul_380);  view_431 = mul_380 = None
        mul_382: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_381, where_13);  mul_381 = None
        sum_109: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [-1], True)
        neg_6: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_13);  where_13 = None
        fma_5: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_382);  neg_6 = sum_109 = mul_382 = None
        view_432: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_5, [256, 512, 512]);  fma_5 = None
        bmm_46: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_323, view_432);  permute_323 = None
        bmm_47: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_432, permute_324);  view_432 = permute_324 = None
        view_433: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_46, [64, 4, 64, 512]);  bmm_46 = None
        view_434: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [64, 4, 512, 64]);  bmm_47 = None
        mul_383: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_433, 0.3535533905932738);  view_433 = None
        permute_325: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_383, [0, 1, 3, 2]);  mul_383 = None
        mul_384: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_434, 0.3535533905932738);  view_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_326: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_88: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        view_435: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_88, [64, 512, 256]);  clone_88 = None
        view_436: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_435, [32768, 256]);  view_435 = None
        permute_71: "f32[256, 256]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_327: "f32[256, 256]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_70: "f32[32768, 256]" = torch.ops.aten.mm.default(view_436, permute_327);  permute_327 = None
        permute_328: "f32[256, 32768]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_71: "f32[256, 256]" = torch.ops.aten.mm.default(permute_328, view_134);  permute_328 = None
        sum_110: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        view_437: "f32[256]" = torch.ops.aten.reshape.default(sum_110, [256]);  sum_110 = None
        view_438: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_70, [64, 512, 256]);  mm_70 = None
        add_141: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_376, view_438);  mul_376 = view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_331: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
        view_439: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_331, [64, 512, 256]);  permute_331 = None
        clone_89: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_439, memory_format = torch.contiguous_format);  view_439 = None
        view_440: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_89, [32768, 256]);  clone_89 = None
        permute_69: "f32[256, 256]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_332: "f32[256, 256]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_72: "f32[32768, 256]" = torch.ops.aten.mm.default(view_440, permute_332);  permute_332 = None
        permute_333: "f32[256, 32768]" = torch.ops.aten.permute.default(view_440, [1, 0])
        mm_73: "f32[256, 256]" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = None
        sum_111: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
        view_441: "f32[256]" = torch.ops.aten.reshape.default(sum_111, [256]);  sum_111 = None
        view_442: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_72, [64, 512, 256]);  mm_72 = None
        add_142: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_141, view_442);  add_141 = view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_336: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_384, [0, 2, 1, 3]);  mul_384 = None
        clone_90: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_443: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_90, [64, 512, 256]);  clone_90 = None
        view_444: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_443, [32768, 256]);  view_443 = None
        permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_337: "f32[256, 256]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_74: "f32[32768, 256]" = torch.ops.aten.mm.default(view_444, permute_337);  permute_337 = None
        permute_338: "f32[256, 32768]" = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_75: "f32[256, 256]" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
        sum_112: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        view_445: "f32[256]" = torch.ops.aten.reshape.default(sum_112, [256]);  sum_112 = None
        view_446: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_74, [64, 512, 256]);  mm_74 = None
        add_143: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_142, view_446);  add_142 = view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_386: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_143, primals_106);  primals_106 = None
        mul_387: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, 256)
        sum_113: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, mul_92);  mul_386 = None
        sum_114: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_92, sum_114);  sum_114 = None
        sub_81: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_387, sum_113);  mul_387 = sum_113 = None
        sub_82: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_81, mul_389);  sub_81 = mul_389 = None
        mul_390: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_82);  div_27 = sub_82 = None
        mul_391: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_143, mul_92);  mul_92 = None
        sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_116: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_392: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_393: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_390, mul_392);  mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_447: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_393, [32768, 256]);  mul_393 = None
        permute_66: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_341: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_76: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_447, permute_341);  permute_341 = None
        permute_342: "f32[256, 32768]" = torch.ops.aten.permute.default(view_447, [1, 0])
        mm_77: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
        sum_117: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        view_448: "f32[256]" = torch.ops.aten.reshape.default(sum_117, [256]);  sum_117 = None
        view_449: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_76, [64, 512, 1024]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_131: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_35, [64, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_88: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
        erf_5: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_395: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_396: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, view_131)
        mul_397: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_21: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_399: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, mul_398);  view_131 = mul_398 = None
        add_145: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_449, add_145);  view_449 = add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_450: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_400, [32768, 1024]);  mul_400 = None
        permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_345: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_78: "f32[32768, 256]" = torch.ops.aten.mm.default(view_450, permute_345);  permute_345 = None
        permute_346: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_450, [1, 0])
        mm_79: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
        sum_118: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        view_451: "f32[1024]" = torch.ops.aten.reshape.default(sum_118, [1024]);  sum_118 = None
        view_452: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_78, [64, 512, 256]);  mm_78 = None
        add_146: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_390, view_452);  mul_390 = view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_402: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_146, primals_100);  primals_100 = None
        mul_403: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_402, 256)
        sum_119: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_402, mul_85);  mul_402 = None
        sum_120: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, sum_120);  sum_120 = None
        sub_84: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_403, sum_119);  mul_403 = sum_119 = None
        sub_85: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_84, mul_405);  sub_84 = mul_405 = None
        mul_406: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_28, sub_85);  div_28 = sub_85 = None
        mul_407: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_146, mul_85);  mul_85 = None
        sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_122: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_408: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_409: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_406, mul_408);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_453: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_409, [32768, 256]);  mul_409 = None
        permute_64: "f32[256, 256]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_349: "f32[256, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_80: "f32[32768, 256]" = torch.ops.aten.mm.default(view_453, permute_349);  permute_349 = None
        permute_350: "f32[256, 32768]" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
        sum_123: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        view_454: "f32[256]" = torch.ops.aten.reshape.default(sum_123, [256]);  sum_123 = None
        view_455: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_80, [64, 512, 256]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_456: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_455, [64, 512, 4, 64]);  view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_353: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_93: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_457: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_93, [256, 512, 64]);  clone_93 = None
        bmm_48: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_354, view_457);  permute_354 = None
        bmm_49: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_457, permute_355);  view_457 = permute_355 = None
        view_458: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [64, 4, 512, 64]);  bmm_48 = None
        view_459: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [64, 4, 512, 512]);  bmm_49 = None
        convert_element_type_21: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_410: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_411: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_459, mul_410);  view_459 = mul_410 = None
        mul_412: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_411, where_11);  mul_411 = None
        sum_124: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_7: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_11);  where_11 = None
        fma_6: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_412);  neg_7 = sum_124 = mul_412 = None
        view_460: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_6, [256, 512, 512]);  fma_6 = None
        bmm_50: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_356, view_460);  permute_356 = None
        bmm_51: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_460, permute_357);  view_460 = permute_357 = None
        view_461: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_50, [64, 4, 64, 512]);  bmm_50 = None
        view_462: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [64, 4, 512, 64]);  bmm_51 = None
        mul_413: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_461, 0.3535533905932738);  view_461 = None
        permute_358: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_413, [0, 1, 3, 2]);  mul_413 = None
        mul_414: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_462, 0.3535533905932738);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_359: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_95: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_463: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_95, [64, 512, 256]);  clone_95 = None
        view_464: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_463, [32768, 256]);  view_463 = None
        permute_60: "f32[256, 256]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_360: "f32[256, 256]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_82: "f32[32768, 256]" = torch.ops.aten.mm.default(view_464, permute_360);  permute_360 = None
        permute_361: "f32[256, 32768]" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_361, view_112);  permute_361 = None
        sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        view_465: "f32[256]" = torch.ops.aten.reshape.default(sum_125, [256]);  sum_125 = None
        view_466: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_82, [64, 512, 256]);  mm_82 = None
        add_147: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_406, view_466);  mul_406 = view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_364: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
        view_467: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_364, [64, 512, 256]);  permute_364 = None
        clone_96: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_467, memory_format = torch.contiguous_format);  view_467 = None
        view_468: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_96, [32768, 256]);  clone_96 = None
        permute_58: "f32[256, 256]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_365: "f32[256, 256]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_84: "f32[32768, 256]" = torch.ops.aten.mm.default(view_468, permute_365);  permute_365 = None
        permute_366: "f32[256, 32768]" = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_85: "f32[256, 256]" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = None
        sum_126: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        view_469: "f32[256]" = torch.ops.aten.reshape.default(sum_126, [256]);  sum_126 = None
        view_470: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_84, [64, 512, 256]);  mm_84 = None
        add_148: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_147, view_470);  add_147 = view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_369: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_414, [0, 2, 1, 3]);  mul_414 = None
        clone_97: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_471: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_97, [64, 512, 256]);  clone_97 = None
        view_472: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_471, [32768, 256]);  view_471 = None
        permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_370: "f32[256, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_86: "f32[32768, 256]" = torch.ops.aten.mm.default(view_472, permute_370);  permute_370 = None
        permute_371: "f32[256, 32768]" = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_87: "f32[256, 256]" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
        sum_127: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        view_473: "f32[256]" = torch.ops.aten.reshape.default(sum_127, [256]);  sum_127 = None
        view_474: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_86, [64, 512, 256]);  mm_86 = None
        add_149: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_148, view_474);  add_148 = view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_416: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_149, primals_90);  primals_90 = None
        mul_417: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_416, 256)
        sum_128: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
        mul_418: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_416, mul_77);  mul_416 = None
        sum_129: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_77, sum_129);  sum_129 = None
        sub_87: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_417, sum_128);  mul_417 = sum_128 = None
        sub_88: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_87, mul_419);  sub_87 = mul_419 = None
        mul_420: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_29, sub_88);  div_29 = sub_88 = None
        mul_421: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_149, mul_77);  mul_77 = None
        sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
        sum_131: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_422: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_423: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_420, mul_422);  mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_475: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_423, [32768, 256]);  mul_423 = None
        permute_55: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_374: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_88: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_475, permute_374);  permute_374 = None
        permute_375: "f32[256, 32768]" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_89: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
        sum_132: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        view_476: "f32[256]" = torch.ops.aten.reshape.default(sum_132, [256]);  sum_132 = None
        view_477: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_88, [64, 512, 1024]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_109: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_29, [64, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_73: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
        erf_4: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_425: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_426: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, view_109)
        mul_427: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_426, -0.5);  mul_426 = None
        exp_22: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_427);  mul_427 = None
        mul_428: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_429: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, mul_428);  view_109 = mul_428 = None
        add_151: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_425, mul_429);  mul_425 = mul_429 = None
        mul_430: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_477, add_151);  view_477 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_430, [32768, 1024]);  mul_430 = None
        permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_378: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_90: "f32[32768, 256]" = torch.ops.aten.mm.default(view_478, permute_378);  permute_378 = None
        permute_379: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_91: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
        sum_133: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        view_479: "f32[1024]" = torch.ops.aten.reshape.default(sum_133, [1024]);  sum_133 = None
        view_480: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_90, [64, 512, 256]);  mm_90 = None
        add_152: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_420, view_480);  mul_420 = view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_432: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_152, primals_84);  primals_84 = None
        mul_433: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_432, 256)
        sum_134: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_432, mul_70);  mul_432 = None
        sum_135: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_70, sum_135);  sum_135 = None
        sub_90: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_433, sum_134);  mul_433 = sum_134 = None
        sub_91: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_90, mul_435);  sub_90 = mul_435 = None
        mul_436: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_91);  div_30 = sub_91 = None
        mul_437: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_152, mul_70);  mul_70 = None
        sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_438: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_439: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_436, mul_438);  mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_439, [32768, 256]);  mul_439 = None
        permute_53: "f32[256, 256]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_382: "f32[256, 256]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_92: "f32[32768, 256]" = torch.ops.aten.mm.default(view_481, permute_382);  permute_382 = None
        permute_383: "f32[256, 32768]" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_93: "f32[256, 256]" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
        sum_138: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        view_482: "f32[256]" = torch.ops.aten.reshape.default(sum_138, [256]);  sum_138 = None
        view_483: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_92, [64, 512, 256]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_484: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_483, [64, 512, 4, 64]);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_386: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_100: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_485: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_100, [256, 512, 64]);  clone_100 = None
        bmm_52: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_387, view_485);  permute_387 = None
        bmm_53: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_485, permute_388);  view_485 = permute_388 = None
        view_486: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [64, 4, 512, 64]);  bmm_52 = None
        view_487: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_53, [64, 4, 512, 512]);  bmm_53 = None
        convert_element_type_24: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_440: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_441: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_487, mul_440);  view_487 = mul_440 = None
        mul_442: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_441, where_9);  mul_441 = None
        sum_139: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_8: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_9);  where_9 = None
        fma_7: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_442);  neg_8 = sum_139 = mul_442 = None
        view_488: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_7, [256, 512, 512]);  fma_7 = None
        bmm_54: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_389, view_488);  permute_389 = None
        bmm_55: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_488, permute_390);  view_488 = permute_390 = None
        view_489: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_54, [64, 4, 64, 512]);  bmm_54 = None
        view_490: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [64, 4, 512, 64]);  bmm_55 = None
        mul_443: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_489, 0.3535533905932738);  view_489 = None
        permute_391: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_443, [0, 1, 3, 2]);  mul_443 = None
        mul_444: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_490, 0.3535533905932738);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_392: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_102: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_491: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_102, [64, 512, 256]);  clone_102 = None
        view_492: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_491, [32768, 256]);  view_491 = None
        permute_49: "f32[256, 256]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_393: "f32[256, 256]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_94: "f32[32768, 256]" = torch.ops.aten.mm.default(view_492, permute_393);  permute_393 = None
        permute_394: "f32[256, 32768]" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_394, view_90);  permute_394 = None
        sum_140: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        view_493: "f32[256]" = torch.ops.aten.reshape.default(sum_140, [256]);  sum_140 = None
        view_494: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_94, [64, 512, 256]);  mm_94 = None
        add_153: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_436, view_494);  mul_436 = view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_397: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
        view_495: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_397, [64, 512, 256]);  permute_397 = None
        clone_103: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_495, memory_format = torch.contiguous_format);  view_495 = None
        view_496: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_103, [32768, 256]);  clone_103 = None
        permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_398: "f32[256, 256]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_96: "f32[32768, 256]" = torch.ops.aten.mm.default(view_496, permute_398);  permute_398 = None
        permute_399: "f32[256, 32768]" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_97: "f32[256, 256]" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = None
        sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        view_497: "f32[256]" = torch.ops.aten.reshape.default(sum_141, [256]);  sum_141 = None
        view_498: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_96, [64, 512, 256]);  mm_96 = None
        add_154: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_153, view_498);  add_153 = view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_402: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_444, [0, 2, 1, 3]);  mul_444 = None
        clone_104: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
        view_499: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_104, [64, 512, 256]);  clone_104 = None
        view_500: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_499, [32768, 256]);  view_499 = None
        permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_403: "f32[256, 256]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_98: "f32[32768, 256]" = torch.ops.aten.mm.default(view_500, permute_403);  permute_403 = None
        permute_404: "f32[256, 32768]" = torch.ops.aten.permute.default(view_500, [1, 0])
        mm_99: "f32[256, 256]" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
        sum_142: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        view_501: "f32[256]" = torch.ops.aten.reshape.default(sum_142, [256]);  sum_142 = None
        view_502: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_98, [64, 512, 256]);  mm_98 = None
        add_155: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_154, view_502);  add_154 = view_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_446: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_155, primals_74);  primals_74 = None
        mul_447: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_446, 256)
        sum_143: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        mul_448: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_446, mul_62);  mul_446 = None
        sum_144: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_62, sum_144);  sum_144 = None
        sub_93: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_447, sum_143);  mul_447 = sum_143 = None
        sub_94: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_93, mul_449);  sub_93 = mul_449 = None
        mul_450: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_94);  div_31 = sub_94 = None
        mul_451: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_155, mul_62);  mul_62 = None
        sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_452: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_453: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_450, mul_452);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_453, [32768, 256]);  mul_453 = None
        permute_44: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_407: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_100: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_503, permute_407);  permute_407 = None
        permute_408: "f32[256, 32768]" = torch.ops.aten.permute.default(view_503, [1, 0])
        mm_101: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
        sum_147: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        view_504: "f32[256]" = torch.ops.aten.reshape.default(sum_147, [256]);  sum_147 = None
        view_505: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_100, [64, 512, 1024]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_87: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_23, [64, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_58: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
        erf_3: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_455: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_456: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, view_87)
        mul_457: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_456, -0.5);  mul_456 = None
        exp_23: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_457);  mul_457 = None
        mul_458: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_459: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, mul_458);  view_87 = mul_458 = None
        add_157: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_455, mul_459);  mul_455 = mul_459 = None
        mul_460: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_505, add_157);  view_505 = add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_506: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_460, [32768, 1024]);  mul_460 = None
        permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_411: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_102: "f32[32768, 256]" = torch.ops.aten.mm.default(view_506, permute_411);  permute_411 = None
        permute_412: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_506, [1, 0])
        mm_103: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
        sum_148: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        view_507: "f32[1024]" = torch.ops.aten.reshape.default(sum_148, [1024]);  sum_148 = None
        view_508: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_102, [64, 512, 256]);  mm_102 = None
        add_158: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_450, view_508);  mul_450 = view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_462: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_158, primals_68);  primals_68 = None
        mul_463: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_462, 256)
        sum_149: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_462, mul_55);  mul_462 = None
        sum_150: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_55, sum_150);  sum_150 = None
        sub_96: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_463, sum_149);  mul_463 = sum_149 = None
        sub_97: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_96, mul_465);  sub_96 = mul_465 = None
        mul_466: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_32, sub_97);  div_32 = sub_97 = None
        mul_467: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_158, mul_55);  mul_55 = None
        sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_152: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_468: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_469: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_466, mul_468);  mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_509: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_469, [32768, 256]);  mul_469 = None
        permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_415: "f32[256, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_104: "f32[32768, 256]" = torch.ops.aten.mm.default(view_509, permute_415);  permute_415 = None
        permute_416: "f32[256, 32768]" = torch.ops.aten.permute.default(view_509, [1, 0])
        mm_105: "f32[256, 256]" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
        sum_153: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        view_510: "f32[256]" = torch.ops.aten.reshape.default(sum_153, [256]);  sum_153 = None
        view_511: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_104, [64, 512, 256]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_512: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_511, [64, 512, 4, 64]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_419: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_107: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_513: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_107, [256, 512, 64]);  clone_107 = None
        bmm_56: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_420, view_513);  permute_420 = None
        bmm_57: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_513, permute_421);  view_513 = permute_421 = None
        view_514: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [64, 4, 512, 64]);  bmm_56 = None
        view_515: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_57, [64, 4, 512, 512]);  bmm_57 = None
        convert_element_type_27: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_470: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_471: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_515, mul_470);  view_515 = mul_470 = None
        mul_472: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_471, where_7);  mul_471 = None
        sum_154: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
        neg_9: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_7);  where_7 = None
        fma_8: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_472);  neg_9 = sum_154 = mul_472 = None
        view_516: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_8, [256, 512, 512]);  fma_8 = None
        bmm_58: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_422, view_516);  permute_422 = None
        bmm_59: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_516, permute_423);  view_516 = permute_423 = None
        view_517: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_58, [64, 4, 64, 512]);  bmm_58 = None
        view_518: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [64, 4, 512, 64]);  bmm_59 = None
        mul_473: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_517, 0.3535533905932738);  view_517 = None
        permute_424: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_473, [0, 1, 3, 2]);  mul_473 = None
        mul_474: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_518, 0.3535533905932738);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_425: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        clone_109: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
        view_519: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_109, [64, 512, 256]);  clone_109 = None
        view_520: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_519, [32768, 256]);  view_519 = None
        permute_38: "f32[256, 256]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_426: "f32[256, 256]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_106: "f32[32768, 256]" = torch.ops.aten.mm.default(view_520, permute_426);  permute_426 = None
        permute_427: "f32[256, 32768]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_107: "f32[256, 256]" = torch.ops.aten.mm.default(permute_427, view_68);  permute_427 = None
        sum_155: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        view_521: "f32[256]" = torch.ops.aten.reshape.default(sum_155, [256]);  sum_155 = None
        view_522: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_106, [64, 512, 256]);  mm_106 = None
        add_159: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_466, view_522);  mul_466 = view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_430: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
        view_523: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_430, [64, 512, 256]);  permute_430 = None
        clone_110: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_523, memory_format = torch.contiguous_format);  view_523 = None
        view_524: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_110, [32768, 256]);  clone_110 = None
        permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_431: "f32[256, 256]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_108: "f32[32768, 256]" = torch.ops.aten.mm.default(view_524, permute_431);  permute_431 = None
        permute_432: "f32[256, 32768]" = torch.ops.aten.permute.default(view_524, [1, 0])
        mm_109: "f32[256, 256]" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = None
        sum_156: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
        view_525: "f32[256]" = torch.ops.aten.reshape.default(sum_156, [256]);  sum_156 = None
        view_526: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_108, [64, 512, 256]);  mm_108 = None
        add_160: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_159, view_526);  add_159 = view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_435: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_474, [0, 2, 1, 3]);  mul_474 = None
        clone_111: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
        view_527: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_111, [64, 512, 256]);  clone_111 = None
        view_528: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_527, [32768, 256]);  view_527 = None
        permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_436: "f32[256, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_110: "f32[32768, 256]" = torch.ops.aten.mm.default(view_528, permute_436);  permute_436 = None
        permute_437: "f32[256, 32768]" = torch.ops.aten.permute.default(view_528, [1, 0])
        mm_111: "f32[256, 256]" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
        sum_157: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        view_529: "f32[256]" = torch.ops.aten.reshape.default(sum_157, [256]);  sum_157 = None
        view_530: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_110, [64, 512, 256]);  mm_110 = None
        add_161: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_160, view_530);  add_160 = view_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_161, primals_58);  primals_58 = None
        mul_477: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_476, 256)
        sum_158: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_476, mul_47);  mul_476 = None
        sum_159: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_47, sum_159);  sum_159 = None
        sub_99: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_477, sum_158);  mul_477 = sum_158 = None
        sub_100: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_99, mul_479);  sub_99 = mul_479 = None
        mul_480: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_100);  div_33 = sub_100 = None
        mul_481: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_161, mul_47);  mul_47 = None
        sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_482: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_483: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_480, mul_482);  mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_531: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_483, [32768, 256]);  mul_483 = None
        permute_33: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_440: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_112: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_531, permute_440);  permute_440 = None
        permute_441: "f32[256, 32768]" = torch.ops.aten.permute.default(view_531, [1, 0])
        mm_113: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
        sum_162: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        view_532: "f32[256]" = torch.ops.aten.reshape.default(sum_162, [256]);  sum_162 = None
        view_533: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_112, [64, 512, 1024]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_65: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_17, [64, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_43: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
        erf_2: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_485: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_486: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, view_65)
        mul_487: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_24: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_489: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, mul_488);  view_65 = mul_488 = None
        add_163: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_533, add_163);  view_533 = add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_534: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_490, [32768, 1024]);  mul_490 = None
        permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        permute_444: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_114: "f32[32768, 256]" = torch.ops.aten.mm.default(view_534, permute_444);  permute_444 = None
        permute_445: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_115: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
        sum_163: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        view_535: "f32[1024]" = torch.ops.aten.reshape.default(sum_163, [1024]);  sum_163 = None
        view_536: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_114, [64, 512, 256]);  mm_114 = None
        add_164: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_480, view_536);  mul_480 = view_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_164, primals_52);  primals_52 = None
        mul_493: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_492, 256)
        sum_164: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_492, mul_40);  mul_492 = None
        sum_165: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_40, sum_165);  sum_165 = None
        sub_102: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_493, sum_164);  mul_493 = sum_164 = None
        sub_103: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_102, mul_495);  sub_102 = mul_495 = None
        mul_496: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_103);  div_34 = sub_103 = None
        mul_497: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_164, mul_40);  mul_40 = None
        sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_498: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_499: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_496, mul_498);  mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_537: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_499, [32768, 256]);  mul_499 = None
        permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_448: "f32[256, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_116: "f32[32768, 256]" = torch.ops.aten.mm.default(view_537, permute_448);  permute_448 = None
        permute_449: "f32[256, 32768]" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_117: "f32[256, 256]" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
        sum_168: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        view_538: "f32[256]" = torch.ops.aten.reshape.default(sum_168, [256]);  sum_168 = None
        view_539: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_116, [64, 512, 256]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_540: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_539, [64, 512, 4, 64]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_452: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_114: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_541: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_114, [256, 512, 64]);  clone_114 = None
        bmm_60: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_453, view_541);  permute_453 = None
        bmm_61: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_541, permute_454);  view_541 = permute_454 = None
        view_542: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [64, 4, 512, 64]);  bmm_60 = None
        view_543: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_61, [64, 4, 512, 512]);  bmm_61 = None
        convert_element_type_30: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_500: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_501: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_543, mul_500);  view_543 = mul_500 = None
        mul_502: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_501, where_5);  mul_501 = None
        sum_169: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_10: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_5);  where_5 = None
        fma_9: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_502);  neg_10 = sum_169 = mul_502 = None
        view_544: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_9, [256, 512, 512]);  fma_9 = None
        bmm_62: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_455, view_544);  permute_455 = None
        bmm_63: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_544, permute_456);  view_544 = permute_456 = None
        view_545: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_62, [64, 4, 64, 512]);  bmm_62 = None
        view_546: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [64, 4, 512, 64]);  bmm_63 = None
        mul_503: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_545, 0.3535533905932738);  view_545 = None
        permute_457: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_503, [0, 1, 3, 2]);  mul_503 = None
        mul_504: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_546, 0.3535533905932738);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_458: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
        clone_116: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_547: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_116, [64, 512, 256]);  clone_116 = None
        view_548: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_547, [32768, 256]);  view_547 = None
        permute_27: "f32[256, 256]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_459: "f32[256, 256]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_118: "f32[32768, 256]" = torch.ops.aten.mm.default(view_548, permute_459);  permute_459 = None
        permute_460: "f32[256, 32768]" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_119: "f32[256, 256]" = torch.ops.aten.mm.default(permute_460, view_46);  permute_460 = None
        sum_170: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549: "f32[256]" = torch.ops.aten.reshape.default(sum_170, [256]);  sum_170 = None
        view_550: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_118, [64, 512, 256]);  mm_118 = None
        add_165: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_496, view_550);  mul_496 = view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_463: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
        view_551: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_463, [64, 512, 256]);  permute_463 = None
        clone_117: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_551, memory_format = torch.contiguous_format);  view_551 = None
        view_552: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_117, [32768, 256]);  clone_117 = None
        permute_25: "f32[256, 256]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_464: "f32[256, 256]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_120: "f32[32768, 256]" = torch.ops.aten.mm.default(view_552, permute_464);  permute_464 = None
        permute_465: "f32[256, 32768]" = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_121: "f32[256, 256]" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = None
        sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
        view_553: "f32[256]" = torch.ops.aten.reshape.default(sum_171, [256]);  sum_171 = None
        view_554: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_120, [64, 512, 256]);  mm_120 = None
        add_166: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_165, view_554);  add_165 = view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_468: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_504, [0, 2, 1, 3]);  mul_504 = None
        clone_118: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_555: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_118, [64, 512, 256]);  clone_118 = None
        view_556: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_555, [32768, 256]);  view_555 = None
        permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_469: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_122: "f32[32768, 256]" = torch.ops.aten.mm.default(view_556, permute_469);  permute_469 = None
        permute_470: "f32[256, 32768]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_123: "f32[256, 256]" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
        sum_172: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        view_557: "f32[256]" = torch.ops.aten.reshape.default(sum_172, [256]);  sum_172 = None
        view_558: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_122, [64, 512, 256]);  mm_122 = None
        add_167: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_166, view_558);  add_166 = view_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_506: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_167, primals_42);  primals_42 = None
        mul_507: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_506, 256)
        sum_173: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
        mul_508: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_506, mul_32);  mul_506 = None
        sum_174: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_32, sum_174);  sum_174 = None
        sub_105: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_507, sum_173);  mul_507 = sum_173 = None
        sub_106: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_105, mul_509);  sub_105 = mul_509 = None
        mul_510: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_35, sub_106);  div_35 = sub_106 = None
        mul_511: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_167, mul_32);  mul_32 = None
        sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
        sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_512: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_513: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_510, mul_512);  mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_559: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_513, [32768, 256]);  mul_513 = None
        permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_473: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_124: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_559, permute_473);  permute_473 = None
        permute_474: "f32[256, 32768]" = torch.ops.aten.permute.default(view_559, [1, 0])
        mm_125: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
        sum_177: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        view_560: "f32[256]" = torch.ops.aten.reshape.default(sum_177, [256]);  sum_177 = None
        view_561: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_124, [64, 512, 1024]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_43: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_11, [64, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_28: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
        erf_1: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_515: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_516: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, view_43)
        mul_517: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
        exp_25: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
        mul_518: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_519: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, mul_518);  view_43 = mul_518 = None
        add_169: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
        mul_520: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_561, add_169);  view_561 = add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_562: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_520, [32768, 1024]);  mul_520 = None
        permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_477: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_126: "f32[32768, 256]" = torch.ops.aten.mm.default(view_562, permute_477);  permute_477 = None
        permute_478: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_127: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
        sum_178: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        view_563: "f32[1024]" = torch.ops.aten.reshape.default(sum_178, [1024]);  sum_178 = None
        view_564: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_126, [64, 512, 256]);  mm_126 = None
        add_170: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_510, view_564);  mul_510 = view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_522: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_170, primals_36);  primals_36 = None
        mul_523: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_522, 256)
        sum_179: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
        mul_524: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_522, mul_25);  mul_522 = None
        sum_180: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
        mul_525: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_25, sum_180);  sum_180 = None
        sub_108: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_523, sum_179);  mul_523 = sum_179 = None
        sub_109: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_108, mul_525);  sub_108 = mul_525 = None
        mul_526: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_109);  div_36 = sub_109 = None
        mul_527: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_170, mul_25);  mul_25 = None
        sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
        sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_528: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_529: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_526, mul_528);  mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_565: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_529, [32768, 256]);  mul_529 = None
        permute_20: "f32[256, 256]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_481: "f32[256, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_128: "f32[32768, 256]" = torch.ops.aten.mm.default(view_565, permute_481);  permute_481 = None
        permute_482: "f32[256, 32768]" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_129: "f32[256, 256]" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
        sum_183: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        view_566: "f32[256]" = torch.ops.aten.reshape.default(sum_183, [256]);  sum_183 = None
        view_567: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_128, [64, 512, 256]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_568: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_567, [64, 512, 4, 64]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_485: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_121: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_569: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_121, [256, 512, 64]);  clone_121 = None
        bmm_64: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_486, view_569);  permute_486 = None
        bmm_65: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_569, permute_487);  view_569 = permute_487 = None
        view_570: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [64, 4, 512, 64]);  bmm_64 = None
        view_571: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, [64, 4, 512, 512]);  bmm_65 = None
        convert_element_type_33: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_530: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_531: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_571, mul_530);  view_571 = mul_530 = None
        mul_532: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_531, where_3);  mul_531 = None
        sum_184: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_11: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_10: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_532);  neg_11 = sum_184 = mul_532 = None
        view_572: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_10, [256, 512, 512]);  fma_10 = None
        bmm_66: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_488, view_572);  permute_488 = None
        bmm_67: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_572, permute_489);  view_572 = permute_489 = None
        view_573: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, [64, 4, 64, 512]);  bmm_66 = None
        view_574: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [64, 4, 512, 64]);  bmm_67 = None
        mul_533: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_573, 0.3535533905932738);  view_573 = None
        permute_490: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_533, [0, 1, 3, 2]);  mul_533 = None
        mul_534: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_574, 0.3535533905932738);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_491: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        clone_123: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_575: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_123, [64, 512, 256]);  clone_123 = None
        view_576: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_575, [32768, 256]);  view_575 = None
        permute_16: "f32[256, 256]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_492: "f32[256, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_130: "f32[32768, 256]" = torch.ops.aten.mm.default(view_576, permute_492);  permute_492 = None
        permute_493: "f32[256, 32768]" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_131: "f32[256, 256]" = torch.ops.aten.mm.default(permute_493, view_24);  permute_493 = None
        sum_185: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        view_577: "f32[256]" = torch.ops.aten.reshape.default(sum_185, [256]);  sum_185 = None
        view_578: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_130, [64, 512, 256]);  mm_130 = None
        add_171: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_526, view_578);  mul_526 = view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_496: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
        view_579: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_496, [64, 512, 256]);  permute_496 = None
        clone_124: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_579, memory_format = torch.contiguous_format);  view_579 = None
        view_580: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_124, [32768, 256]);  clone_124 = None
        permute_14: "f32[256, 256]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_497: "f32[256, 256]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_132: "f32[32768, 256]" = torch.ops.aten.mm.default(view_580, permute_497);  permute_497 = None
        permute_498: "f32[256, 32768]" = torch.ops.aten.permute.default(view_580, [1, 0])
        mm_133: "f32[256, 256]" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = None
        sum_186: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        view_581: "f32[256]" = torch.ops.aten.reshape.default(sum_186, [256]);  sum_186 = None
        view_582: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_132, [64, 512, 256]);  mm_132 = None
        add_172: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_171, view_582);  add_171 = view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_501: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_534, [0, 2, 1, 3]);  mul_534 = None
        clone_125: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_583: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_125, [64, 512, 256]);  clone_125 = None
        view_584: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_583, [32768, 256]);  view_583 = None
        permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_502: "f32[256, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_134: "f32[32768, 256]" = torch.ops.aten.mm.default(view_584, permute_502);  permute_502 = None
        permute_503: "f32[256, 32768]" = torch.ops.aten.permute.default(view_584, [1, 0])
        mm_135: "f32[256, 256]" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
        sum_187: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        view_585: "f32[256]" = torch.ops.aten.reshape.default(sum_187, [256]);  sum_187 = None
        view_586: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_134, [64, 512, 256]);  mm_134 = None
        add_173: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_172, view_586);  add_172 = view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_536: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_173, primals_26);  primals_26 = None
        mul_537: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_536, 256)
        sum_188: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
        mul_538: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_536, mul_17);  mul_536 = None
        sum_189: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, sum_189);  sum_189 = None
        sub_111: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_537, sum_188);  mul_537 = sum_188 = None
        sub_112: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_111, mul_539);  sub_111 = mul_539 = None
        mul_540: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_112);  div_37 = sub_112 = None
        mul_541: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_173, mul_17);  mul_17 = None
        sum_190: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
        sum_191: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_542: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_543: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_540, mul_542);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_587: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_543, [32768, 256]);  mul_543 = None
        permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_506: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_136: "f32[32768, 1024]" = torch.ops.aten.mm.default(view_587, permute_506);  permute_506 = None
        permute_507: "f32[256, 32768]" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_137: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
        sum_192: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        view_588: "f32[256]" = torch.ops.aten.reshape.default(sum_192, [256]);  sum_192 = None
        view_589: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(mm_136, [64, 512, 1024]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_21: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_5, [64, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_13: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
        erf: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_545: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_546: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
        mul_547: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
        exp_26: "f32[64, 512, 1024]" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
        mul_548: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_549: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_548);  view_21 = mul_548 = None
        add_175: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
        mul_550: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_589, add_175);  view_589 = add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_590: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_550, [32768, 1024]);  mul_550 = None
        permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_510: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_138: "f32[32768, 256]" = torch.ops.aten.mm.default(view_590, permute_510);  permute_510 = None
        permute_511: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_139: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
        sum_193: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f32[1024]" = torch.ops.aten.reshape.default(sum_193, [1024]);  sum_193 = None
        view_592: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_138, [64, 512, 256]);  mm_138 = None
        add_176: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_540, view_592);  mul_540 = view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_552: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_176, primals_20);  primals_20 = None
        mul_553: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_552, 256)
        sum_194: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        mul_554: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_552, mul_10);  mul_552 = None
        sum_195: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, sum_195);  sum_195 = None
        sub_114: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_553, sum_194);  mul_553 = sum_194 = None
        sub_115: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_114, mul_555);  sub_114 = mul_555 = None
        mul_556: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div_38, sub_115);  div_38 = sub_115 = None
        mul_557: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_176, mul_10);  mul_10 = None
        sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_197: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_558: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_559: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_556, mul_558);  mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_593: "f32[32768, 256]" = torch.ops.aten.reshape.default(mul_559, [32768, 256]);  mul_559 = None
        permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_514: "f32[256, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_140: "f32[32768, 256]" = torch.ops.aten.mm.default(view_593, permute_514);  permute_514 = None
        permute_515: "f32[256, 32768]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_141: "f32[256, 256]" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
        sum_198: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        view_594: "f32[256]" = torch.ops.aten.reshape.default(sum_198, [256]);  sum_198 = None
        view_595: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_140, [64, 512, 256]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_596: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_595, [64, 512, 4, 64]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_518: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_128: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_597: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_128, [256, 512, 64]);  clone_128 = None
        bmm_68: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_597);  permute_519 = None
        bmm_69: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_597, permute_520);  view_597 = permute_520 = None
        view_598: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [64, 4, 512, 64]);  bmm_68 = None
        view_599: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, [64, 4, 512, 512]);  bmm_69 = None
        convert_element_type_36: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_560: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_561: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_599, mul_560);  view_599 = mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[64, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [64, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  expand_2 = full_default = None
        view_13: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm, [64, 4, 512, 512]);  bmm = None
        add_6: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = where = None
        sub_1: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_2: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        mul_562: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_561, where_1);  mul_561 = None
        sum_199: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [-1], True)
        neg_12: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_11: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_12, sum_199, mul_562);  neg_12 = sum_199 = mul_562 = None
        view_600: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_11, [256, 512, 512]);  fma_11 = None
        bmm_70: "f32[256, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_600);  permute_521 = None
        bmm_71: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_600, permute_522);  view_600 = permute_522 = None
        view_601: "f32[64, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, [64, 4, 64, 512]);  bmm_70 = None
        view_602: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [64, 4, 512, 64]);  bmm_71 = None
        mul_563: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_601, 0.3535533905932738);  view_601 = None
        permute_523: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(mul_563, [0, 1, 3, 2]);  mul_563 = None
        mul_564: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_602, 0.3535533905932738);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_524: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        clone_130: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        view_603: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_130, [64, 512, 256]);  clone_130 = None
        view_604: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_603, [32768, 256]);  view_603 = None
        permute_5: "f32[256, 256]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_525: "f32[256, 256]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_142: "f32[32768, 256]" = torch.ops.aten.mm.default(view_604, permute_525);  permute_525 = None
        permute_526: "f32[256, 32768]" = torch.ops.aten.permute.default(view_604, [1, 0])
        mm_143: "f32[256, 256]" = torch.ops.aten.mm.default(permute_526, view_2);  permute_526 = None
        sum_200: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
        view_605: "f32[256]" = torch.ops.aten.reshape.default(sum_200, [256]);  sum_200 = None
        view_606: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_142, [64, 512, 256]);  mm_142 = None
        add_177: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_556, view_606);  mul_556 = view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_529: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
        view_607: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(permute_529, [64, 512, 256]);  permute_529 = None
        clone_131: "f32[64, 512, 256]" = torch.ops.aten.clone.default(view_607, memory_format = torch.contiguous_format);  view_607 = None
        view_608: "f32[32768, 256]" = torch.ops.aten.reshape.default(clone_131, [32768, 256]);  clone_131 = None
        permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_530: "f32[256, 256]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_144: "f32[32768, 256]" = torch.ops.aten.mm.default(view_608, permute_530);  permute_530 = None
        permute_531: "f32[256, 32768]" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_145: "f32[256, 256]" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = None
        sum_201: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
        view_609: "f32[256]" = torch.ops.aten.reshape.default(sum_201, [256]);  sum_201 = None
        view_610: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_144, [64, 512, 256]);  mm_144 = None
        add_178: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_177, view_610);  add_177 = view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_534: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(mul_564, [0, 2, 1, 3]);  mul_564 = None
        clone_132: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
        view_611: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_132, [64, 512, 256]);  clone_132 = None
        view_612: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_611, [32768, 256]);  view_611 = None
        permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_535: "f32[256, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_146: "f32[32768, 256]" = torch.ops.aten.mm.default(view_612, permute_535);  permute_535 = None
        permute_536: "f32[256, 32768]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_147: "f32[256, 256]" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
        sum_202: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        view_613: "f32[256]" = torch.ops.aten.reshape.default(sum_202, [256]);  sum_202 = None
        view_614: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(mm_146, [64, 512, 256]);  mm_146 = None
        add_179: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(add_178, view_614);  add_178 = view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        view_615: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_179, [32768, 256]);  add_179 = None
        permute: "f32[128, 256]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_539: "f32[256, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_148: "f32[32768, 128]" = torch.ops.aten.mm.default(view_615, permute_539);  permute_539 = None
        permute_540: "f32[256, 32768]" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_149: "f32[256, 128]" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
        sum_203: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        view_616: "f32[256]" = torch.ops.aten.reshape.default(sum_203, [256]);  sum_203 = None
        view_617: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(mm_148, [64, 512, 128]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:116 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_37: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_565: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_566: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_617, mul_565);  view_617 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_568: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_566, primals_8);  primals_8 = None
        mul_569: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_568, 128)
        sum_204: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True)
        mul_570: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_568, mul);  mul_568 = None
        sum_205: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True);  mul_570 = None
        mul_571: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul, sum_205);  sum_205 = None
        sub_117: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_569, sum_204);  mul_569 = sum_204 = None
        sub_118: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub_117, mul_571);  sub_117 = mul_571 = None
        mul_572: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_118);  div_39 = sub_118 = None
        mul_573: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_566, mul);  mul = None
        sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 1]);  mul_573 = None
        sum_207: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1]);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        sum_208: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_572, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_12: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_12, -1);  eq_12 = None
        where_28: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, sum_208);  unsqueeze_5 = sum_208 = None
        full_default_42: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default_42, [primals_3], where_28, True);  full_default_42 = primals_3 = where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[64, 512]" = torch.ops.aten.expand.default(gather, [64, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_13: "b8[64, 512]" = torch.ops.aten.eq.Scalar(expand_1, -1)
        unsqueeze_6: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_13, -1);  eq_13 = None
        where_29: "f32[64, 512, 128]" = torch.ops.aten.where.self(unsqueeze_6, full_default_1, mul_572);  unsqueeze_6 = None
        full_default_44: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_44, [expand_1], where_29, True);  full_default_44 = expand_1 = where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_14: "b8[64, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_7: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_14, -1);  eq_14 = None
        where_30: "f32[64, 512, 128]" = torch.ops.aten.where.self(unsqueeze_7, full_default_1, mul_572);  unsqueeze_7 = full_default_1 = mul_572 = None
        full_default_46: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[30522, 128]" = torch.ops.aten.index_put.default(full_default_46, [primals_2], where_30, True);  full_default_46 = primals_2 = where_30 = None
        add_180: "f32[30522, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_2);  mm_1 = index_put_2 = None
        return (None, None, None, None, add_180, index_put_1, index_put, sum_206, sum_207, mm_149, view_616, mm_147, view_613, mm_145, view_609, mm_143, view_605, mm_141, view_594, sum_196, sum_197, mm_139, view_591, mm_137, view_588, sum_190, sum_191, mm_135, view_585, mm_133, view_581, mm_131, view_577, mm_129, view_566, sum_181, sum_182, mm_127, view_563, mm_125, view_560, sum_175, sum_176, mm_123, view_557, mm_121, view_553, mm_119, view_549, mm_117, view_538, sum_166, sum_167, mm_115, view_535, mm_113, view_532, sum_160, sum_161, mm_111, view_529, mm_109, view_525, mm_107, view_521, mm_105, view_510, sum_151, sum_152, mm_103, view_507, mm_101, view_504, sum_145, sum_146, mm_99, view_501, mm_97, view_497, mm_95, view_493, mm_93, view_482, sum_136, sum_137, mm_91, view_479, mm_89, view_476, sum_130, sum_131, mm_87, view_473, mm_85, view_469, mm_83, view_465, mm_81, view_454, sum_121, sum_122, mm_79, view_451, mm_77, view_448, sum_115, sum_116, mm_75, view_445, mm_73, view_441, mm_71, view_437, mm_69, view_426, sum_106, sum_107, mm_67, view_423, mm_65, view_420, sum_100, sum_101, mm_63, view_417, mm_61, view_413, mm_59, view_409, mm_57, view_398, sum_91, sum_92, mm_55, view_395, mm_53, view_392, sum_85, sum_86, mm_51, view_389, mm_49, view_385, mm_47, view_381, mm_45, view_370, sum_76, sum_77, mm_43, view_367, mm_41, view_364, sum_70, sum_71, mm_39, view_361, mm_37, view_357, mm_35, view_353, mm_33, view_342, sum_61, sum_62, mm_31, view_339, mm_29, view_336, sum_55, sum_56, mm_27, view_333, mm_25, view_329, mm_23, view_325, mm_21, view_314, sum_46, sum_47, mm_19, view_311, mm_17, view_308, sum_40, sum_41, mm_15, view_305, mm_13, view_301, mm_11, view_297, mm_9, view_286, sum_31, sum_32, mm_7, view_283, mm_5, view_280, sum_25, sum_26, mm_3, view_277, sum_20, sum_21, view_274)
