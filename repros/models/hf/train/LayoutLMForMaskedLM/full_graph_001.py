class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "f32[30522, 768]", primals_3: "i64[1, 512]", primals_10: "f32[768]", primals_12: "f32[768, 768]", primals_14: "f32[768, 768]", primals_16: "f32[768, 768]", primals_18: "f32[768, 768]", primals_20: "f32[768]", primals_22: "f32[3072, 768]", primals_24: "f32[768, 3072]", primals_26: "f32[768]", primals_28: "f32[768, 768]", primals_30: "f32[768, 768]", primals_32: "f32[768, 768]", primals_34: "f32[768, 768]", primals_36: "f32[768]", primals_38: "f32[3072, 768]", primals_40: "f32[768, 3072]", primals_42: "f32[768]", primals_44: "f32[768, 768]", primals_46: "f32[768, 768]", primals_48: "f32[768, 768]", primals_50: "f32[768, 768]", primals_52: "f32[768]", primals_54: "f32[3072, 768]", primals_56: "f32[768, 3072]", primals_58: "f32[768]", primals_60: "f32[768, 768]", primals_62: "f32[768, 768]", primals_64: "f32[768, 768]", primals_66: "f32[768, 768]", primals_68: "f32[768]", primals_70: "f32[3072, 768]", primals_72: "f32[768, 3072]", primals_74: "f32[768]", primals_76: "f32[768, 768]", primals_78: "f32[768, 768]", primals_80: "f32[768, 768]", primals_82: "f32[768, 768]", primals_84: "f32[768]", primals_86: "f32[3072, 768]", primals_88: "f32[768, 3072]", primals_90: "f32[768]", primals_92: "f32[768, 768]", primals_94: "f32[768, 768]", primals_96: "f32[768, 768]", primals_98: "f32[768, 768]", primals_100: "f32[768]", primals_102: "f32[3072, 768]", primals_104: "f32[768, 3072]", primals_106: "f32[768]", primals_108: "f32[768, 768]", primals_110: "f32[768, 768]", primals_112: "f32[768, 768]", primals_114: "f32[768, 768]", primals_116: "f32[768]", primals_118: "f32[3072, 768]", primals_120: "f32[768, 3072]", primals_122: "f32[768]", primals_124: "f32[768, 768]", primals_126: "f32[768, 768]", primals_128: "f32[768, 768]", primals_130: "f32[768, 768]", primals_132: "f32[768]", primals_134: "f32[3072, 768]", primals_136: "f32[768, 3072]", primals_138: "f32[768]", primals_140: "f32[768, 768]", primals_142: "f32[768, 768]", primals_144: "f32[768, 768]", primals_146: "f32[768, 768]", primals_148: "f32[768]", primals_150: "f32[3072, 768]", primals_152: "f32[768, 3072]", primals_154: "f32[768]", primals_156: "f32[768, 768]", primals_158: "f32[768, 768]", primals_160: "f32[768, 768]", primals_162: "f32[768, 768]", primals_164: "f32[768]", primals_166: "f32[3072, 768]", primals_168: "f32[768, 3072]", primals_170: "f32[768]", primals_172: "f32[768, 768]", primals_174: "f32[768, 768]", primals_176: "f32[768, 768]", primals_178: "f32[768, 768]", primals_180: "f32[768]", primals_182: "f32[3072, 768]", primals_184: "f32[768, 3072]", primals_186: "f32[768]", primals_188: "f32[768, 768]", primals_190: "f32[768, 768]", primals_192: "f32[768, 768]", primals_194: "f32[768, 768]", primals_196: "f32[768]", primals_198: "f32[3072, 768]", primals_200: "f32[768, 3072]", primals_202: "f32[768]", primals_206: "f32[768, 768]", primals_208: "f32[768]", primals_211: "i64[32, 512]", full_default: "i64[32, 512]", select: "i64[32, 512]", select_1: "i64[32, 512]", select_2: "i64[32, 512]", select_3: "i64[32, 512]", sub_1: "i64[32, 512]", sub_2: "i64[32, 512]", mul_1: "f32[32, 512, 768]", gt: "b8[32, 512, 768]", view: "f32[16384, 768]", clone_default_33: "f32[32, 12, 512, 64]", clone_default_34: "f32[32, 12, 512, 64]", clone_default_35: "f32[32, 12, 512, 64]", getitem_129: "f32[32, 12, 512, 64]", getitem_130: "f32[32, 12, 512]", getitem_131: "i64[]", getitem_132: "i64[]", view_16: "f32[16384, 768]", gt_2: "b8[32, 512, 768]", mul_10: "f32[32, 512, 768]", view_18: "f32[16384, 768]", addmm_4: "f32[16384, 3072]", view_20: "f32[16384, 3072]", gt_3: "b8[32, 512, 768]", mul_17: "f32[32, 512, 768]", view_22: "f32[16384, 768]", clone_default_30: "f32[32, 12, 512, 64]", clone_default_31: "f32[32, 12, 512, 64]", clone_default_32: "f32[32, 12, 512, 64]", getitem_122: "f32[32, 12, 512, 64]", getitem_123: "f32[32, 12, 512]", getitem_124: "i64[]", getitem_125: "i64[]", view_38: "f32[16384, 768]", gt_5: "b8[32, 512, 768]", mul_24: "f32[32, 512, 768]", view_40: "f32[16384, 768]", addmm_10: "f32[16384, 3072]", view_42: "f32[16384, 3072]", gt_6: "b8[32, 512, 768]", mul_31: "f32[32, 512, 768]", view_44: "f32[16384, 768]", clone_default_27: "f32[32, 12, 512, 64]", clone_default_28: "f32[32, 12, 512, 64]", clone_default_29: "f32[32, 12, 512, 64]", getitem_115: "f32[32, 12, 512, 64]", getitem_116: "f32[32, 12, 512]", getitem_117: "i64[]", getitem_118: "i64[]", view_60: "f32[16384, 768]", gt_8: "b8[32, 512, 768]", mul_38: "f32[32, 512, 768]", view_62: "f32[16384, 768]", addmm_16: "f32[16384, 3072]", view_64: "f32[16384, 3072]", gt_9: "b8[32, 512, 768]", mul_45: "f32[32, 512, 768]", view_66: "f32[16384, 768]", clone_default_24: "f32[32, 12, 512, 64]", clone_default_25: "f32[32, 12, 512, 64]", clone_default_26: "f32[32, 12, 512, 64]", getitem_108: "f32[32, 12, 512, 64]", getitem_109: "f32[32, 12, 512]", getitem_110: "i64[]", getitem_111: "i64[]", view_82: "f32[16384, 768]", gt_11: "b8[32, 512, 768]", mul_52: "f32[32, 512, 768]", view_84: "f32[16384, 768]", addmm_22: "f32[16384, 3072]", view_86: "f32[16384, 3072]", gt_12: "b8[32, 512, 768]", mul_59: "f32[32, 512, 768]", view_88: "f32[16384, 768]", clone_default_21: "f32[32, 12, 512, 64]", clone_default_22: "f32[32, 12, 512, 64]", clone_default_23: "f32[32, 12, 512, 64]", getitem_101: "f32[32, 12, 512, 64]", getitem_102: "f32[32, 12, 512]", getitem_103: "i64[]", getitem_104: "i64[]", view_104: "f32[16384, 768]", gt_14: "b8[32, 512, 768]", mul_66: "f32[32, 512, 768]", view_106: "f32[16384, 768]", addmm_28: "f32[16384, 3072]", view_108: "f32[16384, 3072]", gt_15: "b8[32, 512, 768]", mul_73: "f32[32, 512, 768]", view_110: "f32[16384, 768]", clone_default_18: "f32[32, 12, 512, 64]", clone_default_19: "f32[32, 12, 512, 64]", clone_default_20: "f32[32, 12, 512, 64]", getitem_94: "f32[32, 12, 512, 64]", getitem_95: "f32[32, 12, 512]", getitem_96: "i64[]", getitem_97: "i64[]", view_126: "f32[16384, 768]", gt_17: "b8[32, 512, 768]", mul_80: "f32[32, 512, 768]", view_128: "f32[16384, 768]", addmm_34: "f32[16384, 3072]", view_130: "f32[16384, 3072]", gt_18: "b8[32, 512, 768]", mul_87: "f32[32, 512, 768]", view_132: "f32[16384, 768]", clone_default_15: "f32[32, 12, 512, 64]", clone_default_16: "f32[32, 12, 512, 64]", clone_default_17: "f32[32, 12, 512, 64]", getitem_87: "f32[32, 12, 512, 64]", getitem_88: "f32[32, 12, 512]", getitem_89: "i64[]", getitem_90: "i64[]", view_148: "f32[16384, 768]", gt_20: "b8[32, 512, 768]", mul_94: "f32[32, 512, 768]", view_150: "f32[16384, 768]", addmm_40: "f32[16384, 3072]", view_152: "f32[16384, 3072]", gt_21: "b8[32, 512, 768]", mul_101: "f32[32, 512, 768]", view_154: "f32[16384, 768]", clone_default_12: "f32[32, 12, 512, 64]", clone_default_13: "f32[32, 12, 512, 64]", clone_default_14: "f32[32, 12, 512, 64]", getitem_80: "f32[32, 12, 512, 64]", getitem_81: "f32[32, 12, 512]", getitem_82: "i64[]", getitem_83: "i64[]", view_170: "f32[16384, 768]", gt_23: "b8[32, 512, 768]", mul_108: "f32[32, 512, 768]", view_172: "f32[16384, 768]", addmm_46: "f32[16384, 3072]", view_174: "f32[16384, 3072]", gt_24: "b8[32, 512, 768]", mul_115: "f32[32, 512, 768]", view_176: "f32[16384, 768]", clone_default_9: "f32[32, 12, 512, 64]", clone_default_10: "f32[32, 12, 512, 64]", clone_default_11: "f32[32, 12, 512, 64]", getitem_73: "f32[32, 12, 512, 64]", getitem_74: "f32[32, 12, 512]", getitem_75: "i64[]", getitem_76: "i64[]", view_192: "f32[16384, 768]", gt_26: "b8[32, 512, 768]", mul_122: "f32[32, 512, 768]", view_194: "f32[16384, 768]", addmm_52: "f32[16384, 3072]", view_196: "f32[16384, 3072]", gt_27: "b8[32, 512, 768]", mul_129: "f32[32, 512, 768]", view_198: "f32[16384, 768]", clone_default_6: "f32[32, 12, 512, 64]", clone_default_7: "f32[32, 12, 512, 64]", clone_default_8: "f32[32, 12, 512, 64]", getitem_66: "f32[32, 12, 512, 64]", getitem_67: "f32[32, 12, 512]", getitem_68: "i64[]", getitem_69: "i64[]", view_214: "f32[16384, 768]", gt_29: "b8[32, 512, 768]", mul_136: "f32[32, 512, 768]", view_216: "f32[16384, 768]", addmm_58: "f32[16384, 3072]", view_218: "f32[16384, 3072]", gt_30: "b8[32, 512, 768]", mul_143: "f32[32, 512, 768]", view_220: "f32[16384, 768]", clone_default_3: "f32[32, 12, 512, 64]", clone_default_4: "f32[32, 12, 512, 64]", clone_default_5: "f32[32, 12, 512, 64]", getitem_59: "f32[32, 12, 512, 64]", getitem_60: "f32[32, 12, 512]", getitem_61: "i64[]", getitem_62: "i64[]", view_236: "f32[16384, 768]", gt_32: "b8[32, 512, 768]", mul_150: "f32[32, 512, 768]", view_238: "f32[16384, 768]", addmm_64: "f32[16384, 3072]", view_240: "f32[16384, 3072]", gt_33: "b8[32, 512, 768]", mul_157: "f32[32, 512, 768]", view_242: "f32[16384, 768]", clone_default: "f32[32, 12, 512, 64]", clone_default_1: "f32[32, 12, 512, 64]", clone_default_2: "f32[32, 12, 512, 64]", getitem_52: "f32[32, 12, 512, 64]", getitem_53: "f32[32, 12, 512]", getitem_54: "i64[]", getitem_55: "i64[]", view_258: "f32[16384, 768]", gt_35: "b8[32, 512, 768]", mul_164: "f32[32, 512, 768]", view_260: "f32[16384, 768]", addmm_70: "f32[16384, 3072]", view_262: "f32[16384, 3072]", gt_36: "b8[32, 512, 768]", mul_171: "f32[32, 512, 768]", view_264: "f32[16384, 768]", addmm_73: "f32[16384, 768]", getitem_51: "f32[32, 512, 1]", rsqrt_25: "f32[32, 512, 1]", view_266: "f32[16384, 768]", view_267: "f32[32, 512, 30522]", amax_12: "f32[16384, 1]", log: "f32[16384, 1]", convert_element_type: "f32[]", div_15: "f32[32, 512, 1]", div_16: "f32[32, 512, 1]", div_17: "f32[32, 512, 1]", div_18: "f32[32, 512, 1]", div_19: "f32[32, 512, 1]", div_20: "f32[32, 512, 1]", div_21: "f32[32, 512, 1]", div_22: "f32[32, 512, 1]", div_23: "f32[32, 512, 1]", div_24: "f32[32, 512, 1]", div_25: "f32[32, 512, 1]", div_26: "f32[32, 512, 1]", div_27: "f32[32, 512, 1]", div_28: "f32[32, 512, 1]", div_29: "f32[32, 512, 1]", div_30: "f32[32, 512, 1]", div_31: "f32[32, 512, 1]", div_32: "f32[32, 512, 1]", div_33: "f32[32, 512, 1]", div_34: "f32[32, 512, 1]", div_35: "f32[32, 512, 1]", div_36: "f32[32, 512, 1]", div_37: "f32[32, 512, 1]", div_38: "f32[32, 512, 1]", div_39: "f32[32, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[32, 512, 30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:634 in forward, code: labels.view(-1),
        view_269: "i64[16384]" = torch.ops.aten.reshape.default(primals_211, [-1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        unsqueeze_3: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(view_269, 1);  view_269 = None
        ne_3: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[16384, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_3, full_default_3);  unsqueeze_3 = full_default_3 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[16384, 30522]" = torch.ops.aten.expand.default(where_2, [16384, 30522]);  where_2 = None
        eq_tensor: "b8[16384, 30522]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        where_self: "f32[16384, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[16384, 1]" = torch.ops.aten.where.self(ne_3, div_13, full_default_4);  ne_3 = div_13 = None
        mul_178: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        view_268: "f32[16384, 30522]" = torch.ops.aten.reshape.default(view_267, [-1, 30522]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        sub_41: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
        sub_42: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_41, log);  sub_41 = log = None
        exp_13: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_16: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [1], True)
        mul_179: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_43: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        view_270: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(sub_43, [32, 512, 30522]);  sub_43 = None
        add_109: "f32[32, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_270);  tangents_2 = view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        view_271: "f32[16384, 30522]" = torch.ops.aten.reshape.default(add_109, [16384, 30522]);  add_109 = None
        permute_134: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_135: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        constant_pad_nd_default_1: "f32[16384, 30524]" = torch.ops.aten.constant_pad_nd.default(view_271, [0, 2, 0, 0])
        constant_pad_nd_default_2: "f32[30524, 768]" = torch.ops.aten.constant_pad_nd.default(permute_135, [0, 0, 0, 2]);  permute_135 = None
        mm_default_1: "f32[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        permute_136: "f32[30522, 16384]" = torch.ops.aten.permute.default(view_271, [1, 0])
        constant_pad_nd_default: "f32[30524, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_136, [0, 0, 0, 2]);  permute_136 = None
        mm_default: "f32[30524, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default, view_266);  constant_pad_nd_default = view_266 = None
        slice_tensor: "f32[30522, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None
        sum_17: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        view_272: "f32[30522]" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
        view_273: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_default_1, [32, 512, 768]);  mm_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_181: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_208);  primals_208 = None
        mul_182: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_181, 768)
        sum_18: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        view_265: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_73, [32, 512, 768]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_173: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
        mul_174: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_106: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_175: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, add_106);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_40: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_175, getitem_51);  mul_175 = getitem_51 = None
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = None
        mul_183: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_181, mul_176);  mul_181 = None
        sum_19: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
        mul_184: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, sum_19);  sum_19 = None
        sub_45: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_182, sum_18);  mul_182 = sum_18 = None
        sub_46: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_184);  sub_45 = mul_184 = None
        div_14: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_185: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_46);  div_14 = sub_46 = None
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_176);  mul_176 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1]);  mul_186 = None
        sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_188: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
        mul_189: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, view_265)
        mul_190: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
        exp_14: "f32[32, 512, 768]" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
        mul_191: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_192: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_191);  view_265 = mul_191 = None
        add_111: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
        mul_193: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_185, add_111);  mul_185 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        view_274: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_193, [16384, 768]);  mul_193 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
        permute_139: "f32[768, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_2: "f32[16384, 768]" = torch.ops.aten.mm.default(view_274, permute_139);  permute_139 = None
        permute_140: "f32[768, 16384]" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_140, view_264);  permute_140 = view_264 = None
        sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        view_275: "f32[768]" = torch.ops.aten.reshape.default(sum_22, [768]);  sum_22 = None
        view_276: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [32, 512, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_195: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, primals_202);  primals_202 = None
        mul_196: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, 768)
        sum_23: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, mul_171);  mul_195 = None
        sum_24: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_171, sum_24);  sum_24 = None
        sub_48: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_196, sum_23);  mul_196 = sum_23 = None
        sub_49: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_198);  sub_48 = mul_198 = None
        mul_199: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_49);  div_15 = sub_49 = None
        mul_200: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, mul_171);  mul_171 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 1]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_201: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_202: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, mul_201);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_277: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_202, [16384, 768]);  mul_202 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_4: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_277, permute_143);  permute_143 = None
        permute_144: "f32[768, 16384]" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_144, view_262);  permute_144 = view_262 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
        view_278: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_279: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [32, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_167: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_102: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_204: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
        mul_205: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_206: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_205, -0.5);  mul_205 = None
        exp_15: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_206);  mul_206 = None
        mul_207: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_208: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_207);  view_261 = mul_207 = None
        add_113: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_204, mul_208);  mul_204 = mul_208 = None
        mul_209: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_279, add_113);  view_279 = add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_209, [16384, 3072]);  mul_209 = None
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_147: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_6: "f32[16384, 768]" = torch.ops.aten.mm.default(view_280, permute_147);  permute_147 = None
        permute_148: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_148, view_260);  permute_148 = view_260 = None
        sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        view_281: "f32[3072]" = torch.ops.aten.reshape.default(sum_28, [3072]);  sum_28 = None
        view_282: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [32, 512, 768]);  mm_6 = None
        add_114: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_199, view_282);  mul_199 = view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_211: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, primals_196);  primals_196 = None
        mul_212: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_211, 768)
        sum_29: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_211, mul_164);  mul_211 = None
        sum_30: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_164, sum_30);  sum_30 = None
        sub_51: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_212, sum_29);  mul_212 = sum_29 = None
        sub_52: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_214);  sub_51 = mul_214 = None
        mul_215: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_52);  div_16 = sub_52 = None
        mul_216: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_164);  mul_164 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_217: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_218: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_215, mul_217);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_218, [16384, 768]);  mul_218 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_8: "f32[16384, 768]" = torch.ops.aten.mm.default(view_283, permute_151);  permute_151 = None
        permute_152: "f32[768, 16384]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_258);  permute_152 = view_258 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        view_284: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        view_285: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_8, [32, 512, 768]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_286: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_285, [32, 512, 12, 64]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_50: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_50, clone_default, clone_default_1, clone_default_2, None, getitem_52, getitem_53, getitem_54, getitem_55, 0.1, [True, True, True, False], scale = 0.125);  clone_50 = clone_default = clone_default_1 = clone_default_2 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_56: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
        getitem_57: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_58: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_161: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_52: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_293: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_52, [32, 512, 768]);  clone_52 = None
        view_294: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_293, [16384, 768]);  view_293 = None
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_10: "f32[16384, 768]" = torch.ops.aten.mm.default(view_294, permute_162);  permute_162 = None
        permute_163: "f32[768, 16384]" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        view_295: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        view_296: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [32, 512, 768]);  mm_10 = None
        add_115: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_215, view_296);  mul_215 = view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_166: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        view_297: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_166, [32, 512, 768]);  permute_166 = None
        clone_53: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_297, memory_format = torch.contiguous_format);  view_297 = None
        view_298: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_53, [16384, 768]);  clone_53 = None
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_12: "f32[16384, 768]" = torch.ops.aten.mm.default(view_298, permute_167);  permute_167 = None
        permute_168: "f32[768, 16384]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_168, view_242);  permute_168 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        view_299: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        view_300: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_12, [32, 512, 768]);  mm_12 = None
        add_116: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_115, view_300);  add_115 = view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_171: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_54: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_301: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_54, [32, 512, 768]);  clone_54 = None
        view_302: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_301, [16384, 768]);  view_301 = None
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_172: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_14: "f32[16384, 768]" = torch.ops.aten.mm.default(view_302, permute_172);  permute_172 = None
        permute_173: "f32[768, 16384]" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_173, view_242);  permute_173 = view_242 = None
        sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
        view_304: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_14, [32, 512, 768]);  mm_14 = None
        add_117: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_116, view_304);  add_116 = view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_224: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_186);  primals_186 = None
        mul_225: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, 768)
        sum_38: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
        mul_226: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_157);  mul_224 = None
        sum_39: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
        mul_227: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_157, sum_39);  sum_39 = None
        sub_54: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_225, sum_38);  mul_225 = sum_38 = None
        sub_55: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_227);  sub_54 = mul_227 = None
        mul_228: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_55);  div_17 = sub_55 = None
        mul_229: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_157);  mul_157 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_230: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_231: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_228, mul_230);  mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_231, [16384, 768]);  mul_231 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_16: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_305, permute_176);  permute_176 = None
        permute_177: "f32[768, 16384]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_177, view_240);  permute_177 = view_240 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        view_306: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        view_307: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [32, 512, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_153: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_94: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_233: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
        mul_234: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_235: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_234, -0.5);  mul_234 = None
        exp_16: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_235);  mul_235 = None
        mul_236: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_237: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_236);  view_239 = mul_236 = None
        add_119: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_233, mul_237);  mul_233 = mul_237 = None
        mul_238: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_307, add_119);  view_307 = add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_308: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_238, [16384, 3072]);  mul_238 = None
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_180: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_18: "f32[16384, 768]" = torch.ops.aten.mm.default(view_308, permute_180);  permute_180 = None
        permute_181: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_181, view_238);  permute_181 = view_238 = None
        sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
        view_309: "f32[3072]" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
        view_310: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [32, 512, 768]);  mm_18 = None
        add_120: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_228, view_310);  mul_228 = view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_240: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_180);  primals_180 = None
        mul_241: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_240, 768)
        sum_44: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True)
        mul_242: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_240, mul_150);  mul_240 = None
        sum_45: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True);  mul_242 = None
        mul_243: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, sum_45);  sum_45 = None
        sub_57: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_241, sum_44);  mul_241 = sum_44 = None
        sub_58: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_57, mul_243);  sub_57 = mul_243 = None
        mul_244: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_58);  div_18 = sub_58 = None
        mul_245: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_150);  mul_150 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_245, [0, 1]);  mul_245 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_5: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_246: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_247: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_244, mul_246);  mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_311: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_247, [16384, 768]);  mul_247 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_20: "f32[16384, 768]" = torch.ops.aten.mm.default(view_311, permute_184);  permute_184 = None
        permute_185: "f32[768, 16384]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_236);  permute_185 = view_236 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312: "f32[768]" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        view_313: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_20, [32, 512, 768]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_314: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_313, [32, 512, 12, 64]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_188: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_57: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_57, clone_default_3, clone_default_4, clone_default_5, None, getitem_59, getitem_60, getitem_61, getitem_62, 0.1, [True, True, True, False], scale = 0.125);  clone_57 = clone_default_3 = clone_default_4 = clone_default_5 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_63: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
        getitem_64: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_65: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_194: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_65, [0, 2, 1, 3]);  getitem_65 = None
        clone_59: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_321: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_59, [32, 512, 768]);  clone_59 = None
        view_322: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_321, [16384, 768]);  view_321 = None
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_22: "f32[16384, 768]" = torch.ops.aten.mm.default(view_322, permute_195);  permute_195 = None
        permute_196: "f32[768, 16384]" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
        view_323: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
        view_324: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [32, 512, 768]);  mm_22 = None
        add_121: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_244, view_324);  mul_244 = view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_199: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_64, [0, 2, 1, 3]);  getitem_64 = None
        view_325: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_199, [32, 512, 768]);  permute_199 = None
        clone_60: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_325, memory_format = torch.contiguous_format);  view_325 = None
        view_326: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_60, [16384, 768]);  clone_60 = None
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_200: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_24: "f32[16384, 768]" = torch.ops.aten.mm.default(view_326, permute_200);  permute_200 = None
        permute_201: "f32[768, 16384]" = torch.ops.aten.permute.default(view_326, [1, 0])
        mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_201, view_220);  permute_201 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
        view_327: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        view_328: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_24, [32, 512, 768]);  mm_24 = None
        add_122: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_121, view_328);  add_121 = view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_204: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        clone_61: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_329: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_61, [32, 512, 768]);  clone_61 = None
        view_330: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_329, [16384, 768]);  view_329 = None
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_26: "f32[16384, 768]" = torch.ops.aten.mm.default(view_330, permute_205);  permute_205 = None
        permute_206: "f32[768, 16384]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_206, view_220);  permute_206 = view_220 = None
        sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        view_331: "f32[768]" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
        view_332: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_26, [32, 512, 768]);  mm_26 = None
        add_123: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_122, view_332);  add_122 = view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_253: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, primals_170);  primals_170 = None
        mul_254: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, 768)
        sum_53: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
        mul_255: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_143);  mul_253 = None
        sum_54: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
        mul_256: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_143, sum_54);  sum_54 = None
        sub_60: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_254, sum_53);  mul_254 = sum_53 = None
        sub_61: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_256);  sub_60 = mul_256 = None
        mul_257: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_61);  div_19 = sub_61 = None
        mul_258: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_143);  mul_143 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_7: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_259: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_260: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_333: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_260, [16384, 768]);  mul_260 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_28: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_333, permute_209);  permute_209 = None
        permute_210: "f32[768, 16384]" = torch.ops.aten.permute.default(view_333, [1, 0])
        mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_210, view_218);  permute_210 = view_218 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
        view_334: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_335: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [32, 512, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_139: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_86: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_262: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
        mul_263: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_264: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
        exp_17: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
        mul_265: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_266: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_265);  view_217 = mul_265 = None
        add_125: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
        mul_267: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_335, add_125);  view_335 = add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_336: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_267, [16384, 3072]);  mul_267 = None
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        permute_213: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_30: "f32[16384, 768]" = torch.ops.aten.mm.default(view_336, permute_213);  permute_213 = None
        permute_214: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_214, view_216);  permute_214 = view_216 = None
        sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
        view_337: "f32[3072]" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
        view_338: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [32, 512, 768]);  mm_30 = None
        add_126: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_257, view_338);  mul_257 = view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_269: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_164);  primals_164 = None
        mul_270: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
        sum_59: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_136);  mul_269 = None
        sum_60: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_136, sum_60);  sum_60 = None
        sub_63: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_59);  mul_270 = sum_59 = None
        sub_64: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_272);  sub_63 = mul_272 = None
        mul_273: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_64);  div_20 = sub_64 = None
        mul_274: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_136);  mul_136 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_275: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_276: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_275);  mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_339: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_276, [16384, 768]);  mul_276 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_217: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_32: "f32[16384, 768]" = torch.ops.aten.mm.default(view_339, permute_217);  permute_217 = None
        permute_218: "f32[768, 16384]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_214);  permute_218 = view_214 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
        view_340: "f32[768]" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        view_341: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_32, [32, 512, 768]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_342: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_341, [32, 512, 12, 64]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_64: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_64, clone_default_6, clone_default_7, clone_default_8, None, getitem_66, getitem_67, getitem_68, getitem_69, 0.1, [True, True, True, False], scale = 0.125);  clone_64 = clone_default_6 = clone_default_7 = clone_default_8 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_70: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
        getitem_71: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_72: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_227: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None
        clone_66: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_349: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_66, [32, 512, 768]);  clone_66 = None
        view_350: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_349, [16384, 768]);  view_349 = None
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_34: "f32[16384, 768]" = torch.ops.aten.mm.default(view_350, permute_228);  permute_228 = None
        permute_229: "f32[768, 16384]" = torch.ops.aten.permute.default(view_350, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
        view_351: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        view_352: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [32, 512, 768]);  mm_34 = None
        add_127: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_273, view_352);  mul_273 = view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_232: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        view_353: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_232, [32, 512, 768]);  permute_232 = None
        clone_67: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_353, memory_format = torch.contiguous_format);  view_353 = None
        view_354: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_67, [16384, 768]);  clone_67 = None
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_36: "f32[16384, 768]" = torch.ops.aten.mm.default(view_354, permute_233);  permute_233 = None
        permute_234: "f32[768, 16384]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_234, view_198);  permute_234 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True);  view_354 = None
        view_355: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        view_356: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_36, [32, 512, 768]);  mm_36 = None
        add_128: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_127, view_356);  add_127 = view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_237: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_70, [0, 2, 1, 3]);  getitem_70 = None
        clone_68: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_357: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_68, [32, 512, 768]);  clone_68 = None
        view_358: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_357, [16384, 768]);  view_357 = None
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_38: "f32[16384, 768]" = torch.ops.aten.mm.default(view_358, permute_238);  permute_238 = None
        permute_239: "f32[768, 16384]" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_198);  permute_239 = view_198 = None
        sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
        view_359: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
        view_360: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_38, [32, 512, 768]);  mm_38 = None
        add_129: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_128, view_360);  add_128 = view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_282: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_154);  primals_154 = None
        mul_283: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, 768)
        sum_68: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
        mul_284: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, mul_129);  mul_282 = None
        sum_69: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        mul_285: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_129, sum_69);  sum_69 = None
        sub_66: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_283, sum_68);  mul_283 = sum_68 = None
        sub_67: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_285);  sub_66 = mul_285 = None
        mul_286: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_67);  div_21 = sub_67 = None
        mul_287: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_129);  mul_129 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_288: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_289: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_361: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_289, [16384, 768]);  mul_289 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_40: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_361, permute_242);  permute_242 = None
        permute_243: "f32[768, 16384]" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_243, view_196);  permute_243 = view_196 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        view_362: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        view_363: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [32, 512, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_125: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_78: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_291: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
        mul_292: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_293: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_292, -0.5);  mul_292 = None
        exp_18: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_293);  mul_293 = None
        mul_294: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_295: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_294);  view_195 = mul_294 = None
        add_131: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_291, mul_295);  mul_291 = mul_295 = None
        mul_296: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_363, add_131);  view_363 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_364: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_296, [16384, 3072]);  mul_296 = None
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_246: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_42: "f32[16384, 768]" = torch.ops.aten.mm.default(view_364, permute_246);  permute_246 = None
        permute_247: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_247, view_194);  permute_247 = view_194 = None
        sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
        view_365: "f32[3072]" = torch.ops.aten.reshape.default(sum_73, [3072]);  sum_73 = None
        view_366: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [32, 512, 768]);  mm_42 = None
        add_132: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_286, view_366);  mul_286 = view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_298: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_148);  primals_148 = None
        mul_299: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, 768)
        sum_74: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
        mul_300: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, mul_122);  mul_298 = None
        sum_75: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
        mul_301: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, sum_75);  sum_75 = None
        sub_69: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_299, sum_74);  mul_299 = sum_74 = None
        sub_70: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_301);  sub_69 = mul_301 = None
        mul_302: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_70);  div_22 = sub_70 = None
        mul_303: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_122);  mul_122 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_11: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_304: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_305: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, mul_304);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_367: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_305, [16384, 768]);  mul_305 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_44: "f32[16384, 768]" = torch.ops.aten.mm.default(view_367, permute_250);  permute_250 = None
        permute_251: "f32[768, 16384]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_251, view_192);  permute_251 = view_192 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        view_368: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        view_369: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_44, [32, 512, 768]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_370: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_369, [32, 512, 12, 64]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_254: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_71: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_71, clone_default_9, clone_default_10, clone_default_11, None, getitem_73, getitem_74, getitem_75, getitem_76, 0.1, [True, True, True, False], scale = 0.125);  clone_71 = clone_default_9 = clone_default_10 = clone_default_11 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_77: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
        getitem_78: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_79: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_260: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
        clone_73: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_377: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_73, [32, 512, 768]);  clone_73 = None
        view_378: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_377, [16384, 768]);  view_377 = None
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_46: "f32[16384, 768]" = torch.ops.aten.mm.default(view_378, permute_261);  permute_261 = None
        permute_262: "f32[768, 16384]" = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
        view_379: "f32[768]" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
        view_380: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [32, 512, 768]);  mm_46 = None
        add_133: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_302, view_380);  mul_302 = view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_265: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 1, 3]);  getitem_78 = None
        view_381: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_265, [32, 512, 768]);  permute_265 = None
        clone_74: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_381, memory_format = torch.contiguous_format);  view_381 = None
        view_382: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_74, [16384, 768]);  clone_74 = None
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_48: "f32[16384, 768]" = torch.ops.aten.mm.default(view_382, permute_266);  permute_266 = None
        permute_267: "f32[768, 16384]" = torch.ops.aten.permute.default(view_382, [1, 0])
        mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_176);  permute_267 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
        view_383: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        view_384: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_48, [32, 512, 768]);  mm_48 = None
        add_134: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_133, view_384);  add_133 = view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_270: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None
        clone_75: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_385: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_75, [32, 512, 768]);  clone_75 = None
        view_386: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_385, [16384, 768]);  view_385 = None
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_50: "f32[16384, 768]" = torch.ops.aten.mm.default(view_386, permute_271);  permute_271 = None
        permute_272: "f32[768, 16384]" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_272, view_176);  permute_272 = view_176 = None
        sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
        view_387: "f32[768]" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
        view_388: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_50, [32, 512, 768]);  mm_50 = None
        add_135: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_134, view_388);  add_134 = view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_311: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, primals_138);  primals_138 = None
        mul_312: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
        sum_83: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
        mul_313: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_115);  mul_311 = None
        sum_84: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
        mul_314: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, sum_84);  sum_84 = None
        sub_72: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_83);  mul_312 = sum_83 = None
        sub_73: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_314);  sub_72 = mul_314 = None
        mul_315: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_73);  div_23 = sub_73 = None
        mul_316: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, mul_115);  mul_115 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_317: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_318: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, mul_317);  mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_389: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_318, [16384, 768]);  mul_318 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_52: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_389, permute_275);  permute_275 = None
        permute_276: "f32[768, 16384]" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_276, view_174);  permute_276 = view_174 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        view_390: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_391: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_52, [32, 512, 3072]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_111: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_70: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_320: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
        mul_321: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_322: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_321, -0.5);  mul_321 = None
        exp_19: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_322);  mul_322 = None
        mul_323: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_324: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_323);  view_173 = mul_323 = None
        add_137: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_320, mul_324);  mul_320 = mul_324 = None
        mul_325: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_391, add_137);  view_391 = add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_325, [16384, 3072]);  mul_325 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_54: "f32[16384, 768]" = torch.ops.aten.mm.default(view_392, permute_279);  permute_279 = None
        permute_280: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_280, view_172);  permute_280 = view_172 = None
        sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
        view_393: "f32[3072]" = torch.ops.aten.reshape.default(sum_88, [3072]);  sum_88 = None
        view_394: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_54, [32, 512, 768]);  mm_54 = None
        add_138: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_315, view_394);  mul_315 = view_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_327: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_132);  primals_132 = None
        mul_328: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_327, 768)
        sum_89: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_327, mul_108);  mul_327 = None
        sum_90: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_90);  sum_90 = None
        sub_75: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_328, sum_89);  mul_328 = sum_89 = None
        sub_76: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_330);  sub_75 = mul_330 = None
        mul_331: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_76);  div_24 = sub_76 = None
        mul_332: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_108);  mul_108 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_333: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_334: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_331, mul_333);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_395: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_334, [16384, 768]);  mul_334 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_56: "f32[16384, 768]" = torch.ops.aten.mm.default(view_395, permute_283);  permute_283 = None
        permute_284: "f32[768, 16384]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_284, view_170);  permute_284 = view_170 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        view_396: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        view_397: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_56, [32, 512, 768]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_398: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_397, [32, 512, 12, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_78: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_78, clone_default_12, clone_default_13, clone_default_14, None, getitem_80, getitem_81, getitem_82, getitem_83, 0.1, [True, True, True, False], scale = 0.125);  clone_78 = clone_default_12 = clone_default_13 = clone_default_14 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_84: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
        getitem_85: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_86: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_293: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3]);  getitem_86 = None
        clone_80: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_405: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_80, [32, 512, 768]);  clone_80 = None
        view_406: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_405, [16384, 768]);  view_405 = None
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_58: "f32[16384, 768]" = torch.ops.aten.mm.default(view_406, permute_294);  permute_294 = None
        permute_295: "f32[768, 16384]" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        view_407: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
        view_408: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_58, [32, 512, 768]);  mm_58 = None
        add_139: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_331, view_408);  mul_331 = view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_298: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_409: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_298, [32, 512, 768]);  permute_298 = None
        clone_81: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_409, memory_format = torch.contiguous_format);  view_409 = None
        view_410: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_81, [16384, 768]);  clone_81 = None
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_60: "f32[16384, 768]" = torch.ops.aten.mm.default(view_410, permute_299);  permute_299 = None
        permute_300: "f32[768, 16384]" = torch.ops.aten.permute.default(view_410, [1, 0])
        mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_154);  permute_300 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_410, [0], True);  view_410 = None
        view_411: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        view_412: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_60, [32, 512, 768]);  mm_60 = None
        add_140: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_412);  add_139 = view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_303: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None
        clone_82: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_413: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_82, [32, 512, 768]);  clone_82 = None
        view_414: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_413, [16384, 768]);  view_413 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_304: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_62: "f32[16384, 768]" = torch.ops.aten.mm.default(view_414, permute_304);  permute_304 = None
        permute_305: "f32[768, 16384]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_305, view_154);  permute_305 = view_154 = None
        sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
        view_415: "f32[768]" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        view_416: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_62, [32, 512, 768]);  mm_62 = None
        add_141: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_140, view_416);  add_140 = view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_340: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_122);  primals_122 = None
        mul_341: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, 768)
        sum_98: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
        mul_342: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, mul_101);  mul_340 = None
        sum_99: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
        mul_343: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_101, sum_99);  sum_99 = None
        sub_78: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_341, sum_98);  mul_341 = sum_98 = None
        sub_79: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_343);  sub_78 = mul_343 = None
        mul_344: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_79);  div_25 = sub_79 = None
        mul_345: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_101);  mul_101 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_346: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_347: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_344, mul_346);  mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_417: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_347, [16384, 768]);  mul_347 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_64: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_417, permute_308);  permute_308 = None
        permute_309: "f32[768, 16384]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_309, view_152);  permute_309 = view_152 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
        view_418: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        view_419: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_64, [32, 512, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_97: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_62: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_349: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_350: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_351: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_350, -0.5);  mul_350 = None
        exp_20: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_351);  mul_351 = None
        mul_352: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_353: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_352);  view_151 = mul_352 = None
        add_143: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_349, mul_353);  mul_349 = mul_353 = None
        mul_354: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_419, add_143);  view_419 = add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_420: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_354, [16384, 3072]);  mul_354 = None
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_312: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_66: "f32[16384, 768]" = torch.ops.aten.mm.default(view_420, permute_312);  permute_312 = None
        permute_313: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_313, view_150);  permute_313 = view_150 = None
        sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
        view_421: "f32[3072]" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
        view_422: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_66, [32, 512, 768]);  mm_66 = None
        add_144: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_344, view_422);  mul_344 = view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_356: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_144, primals_116);  primals_116 = None
        mul_357: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, 768)
        sum_104: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, mul_94);  mul_356 = None
        sum_105: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, sum_105);  sum_105 = None
        sub_81: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_357, sum_104);  mul_357 = sum_104 = None
        sub_82: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_81, mul_359);  sub_81 = mul_359 = None
        mul_360: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_82);  div_26 = sub_82 = None
        mul_361: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_144, mul_94);  mul_94 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_362: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_363: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_360, mul_362);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_423: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_363, [16384, 768]);  mul_363 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_68: "f32[16384, 768]" = torch.ops.aten.mm.default(view_423, permute_316);  permute_316 = None
        permute_317: "f32[768, 16384]" = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_317, view_148);  permute_317 = view_148 = None
        sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
        view_424: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        view_425: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_68, [32, 512, 768]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_426: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_425, [32, 512, 12, 64]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_320: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_85: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_85, clone_default_15, clone_default_16, clone_default_17, None, getitem_87, getitem_88, getitem_89, getitem_90, 0.1, [True, True, True, False], scale = 0.125);  clone_85 = clone_default_15 = clone_default_16 = clone_default_17 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_91: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
        getitem_92: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_93: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_326: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        clone_87: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        view_433: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_87, [32, 512, 768]);  clone_87 = None
        view_434: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_433, [16384, 768]);  view_433 = None
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_70: "f32[16384, 768]" = torch.ops.aten.mm.default(view_434, permute_327);  permute_327 = None
        permute_328: "f32[768, 16384]" = torch.ops.aten.permute.default(view_434, [1, 0])
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
        sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
        view_435: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
        view_436: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_70, [32, 512, 768]);  mm_70 = None
        add_145: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_360, view_436);  mul_360 = view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_331: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None
        view_437: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_331, [32, 512, 768]);  permute_331 = None
        clone_88: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_437, memory_format = torch.contiguous_format);  view_437 = None
        view_438: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_88, [16384, 768]);  clone_88 = None
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_72: "f32[16384, 768]" = torch.ops.aten.mm.default(view_438, permute_332);  permute_332 = None
        permute_333: "f32[768, 16384]" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_132);  permute_333 = None
        sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
        view_439: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        view_440: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_72, [32, 512, 768]);  mm_72 = None
        add_146: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_145, view_440);  add_145 = view_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_336: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_91, [0, 2, 1, 3]);  getitem_91 = None
        clone_89: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_441: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_89, [32, 512, 768]);  clone_89 = None
        view_442: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_441, [16384, 768]);  view_441 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_337: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_74: "f32[16384, 768]" = torch.ops.aten.mm.default(view_442, permute_337);  permute_337 = None
        permute_338: "f32[768, 16384]" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_338, view_132);  permute_338 = view_132 = None
        sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
        view_443: "f32[768]" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
        view_444: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_74, [32, 512, 768]);  mm_74 = None
        add_147: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_146, view_444);  add_146 = view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_369: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_147, primals_106);  primals_106 = None
        mul_370: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, 768)
        sum_113: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
        mul_371: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, mul_87);  mul_369 = None
        sum_114: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
        mul_372: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_87, sum_114);  sum_114 = None
        sub_84: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_370, sum_113);  mul_370 = sum_113 = None
        sub_85: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_372);  sub_84 = mul_372 = None
        mul_373: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_85);  div_27 = sub_85 = None
        mul_374: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_87);  mul_87 = None
        sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_375: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_376: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, mul_375);  mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_445: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_376, [16384, 768]);  mul_376 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_341: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_76: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_445, permute_341);  permute_341 = None
        permute_342: "f32[768, 16384]" = torch.ops.aten.permute.default(view_445, [1, 0])
        mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_342, view_130);  permute_342 = view_130 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
        view_446: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_447: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_76, [32, 512, 3072]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_83: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_54: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_378: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
        mul_379: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_380: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_379, -0.5);  mul_379 = None
        exp_21: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_380);  mul_380 = None
        mul_381: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_382: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_381);  view_129 = mul_381 = None
        add_149: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_378, mul_382);  mul_378 = mul_382 = None
        mul_383: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_149);  view_447 = add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_448: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_383, [16384, 3072]);  mul_383 = None
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_345: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_78: "f32[16384, 768]" = torch.ops.aten.mm.default(view_448, permute_345);  permute_345 = None
        permute_346: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_346, view_128);  permute_346 = view_128 = None
        sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
        view_449: "f32[3072]" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
        view_450: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_78, [32, 512, 768]);  mm_78 = None
        add_150: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_373, view_450);  mul_373 = view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_385: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_100);  primals_100 = None
        mul_386: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_385, 768)
        sum_119: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True)
        mul_387: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_385, mul_80);  mul_385 = None
        sum_120: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True);  mul_387 = None
        mul_388: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_120);  sum_120 = None
        sub_87: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_386, sum_119);  mul_386 = sum_119 = None
        sub_88: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_388);  sub_87 = mul_388 = None
        mul_389: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_88);  div_28 = sub_88 = None
        mul_390: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_80);  mul_80 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1]);  mul_390 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_391: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_392: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_389, mul_391);  mul_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_451: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_392, [16384, 768]);  mul_392 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_349: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_80: "f32[16384, 768]" = torch.ops.aten.mm.default(view_451, permute_349);  permute_349 = None
        permute_350: "f32[768, 16384]" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_350, view_126);  permute_350 = view_126 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
        view_452: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        view_453: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_80, [32, 512, 768]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_454: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_453, [32, 512, 12, 64]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_353: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_92: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_92, clone_default_18, clone_default_19, clone_default_20, None, getitem_94, getitem_95, getitem_96, getitem_97, 0.1, [True, True, True, False], scale = 0.125);  clone_92 = clone_default_18 = clone_default_19 = clone_default_20 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_98: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
        getitem_99: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_100: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_359: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        clone_94: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_461: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_94, [32, 512, 768]);  clone_94 = None
        view_462: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_461, [16384, 768]);  view_461 = None
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_82: "f32[16384, 768]" = torch.ops.aten.mm.default(view_462, permute_360);  permute_360 = None
        permute_361: "f32[768, 16384]" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
        view_463: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        view_464: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_82, [32, 512, 768]);  mm_82 = None
        add_151: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_389, view_464);  mul_389 = view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_364: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None
        view_465: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_364, [32, 512, 768]);  permute_364 = None
        clone_95: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_465, memory_format = torch.contiguous_format);  view_465 = None
        view_466: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_95, [16384, 768]);  clone_95 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_84: "f32[16384, 768]" = torch.ops.aten.mm.default(view_466, permute_365);  permute_365 = None
        permute_366: "f32[768, 16384]" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_366, view_110);  permute_366 = None
        sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
        view_467: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        view_468: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_84, [32, 512, 768]);  mm_84 = None
        add_152: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_151, view_468);  add_151 = view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_369: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_98, [0, 2, 1, 3]);  getitem_98 = None
        clone_96: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_469: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_96, [32, 512, 768]);  clone_96 = None
        view_470: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_469, [16384, 768]);  view_469 = None
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_370: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_86: "f32[16384, 768]" = torch.ops.aten.mm.default(view_470, permute_370);  permute_370 = None
        permute_371: "f32[768, 16384]" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_371, view_110);  permute_371 = view_110 = None
        sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        view_471: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
        view_472: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_86, [32, 512, 768]);  mm_86 = None
        add_153: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_152, view_472);  add_152 = view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_398: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_90);  primals_90 = None
        mul_399: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, 768)
        sum_128: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
        mul_400: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, mul_73);  mul_398 = None
        sum_129: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
        mul_401: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_129);  sum_129 = None
        sub_90: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_399, sum_128);  mul_399 = sum_128 = None
        sub_91: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_401);  sub_90 = mul_401 = None
        mul_402: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_91);  div_29 = sub_91 = None
        mul_403: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_73);  mul_73 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
        sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_404: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_405: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_473: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_405, [16384, 768]);  mul_405 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_88: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_473, permute_374);  permute_374 = None
        permute_375: "f32[768, 16384]" = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_375, view_108);  permute_375 = view_108 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
        view_474: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        view_475: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_88, [32, 512, 3072]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_69: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_46: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_407: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
        mul_408: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_409: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_408, -0.5);  mul_408 = None
        exp_22: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_409);  mul_409 = None
        mul_410: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_411: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_410);  view_107 = mul_410 = None
        add_155: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_407, mul_411);  mul_407 = mul_411 = None
        mul_412: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_475, add_155);  view_475 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_476: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_412, [16384, 3072]);  mul_412 = None
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_378: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_90: "f32[16384, 768]" = torch.ops.aten.mm.default(view_476, permute_378);  permute_378 = None
        permute_379: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_476, [1, 0])
        mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_379, view_106);  permute_379 = view_106 = None
        sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
        view_477: "f32[3072]" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
        view_478: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_90, [32, 512, 768]);  mm_90 = None
        add_156: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_402, view_478);  mul_402 = view_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_414: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_84);  primals_84 = None
        mul_415: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_414, 768)
        sum_134: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True)
        mul_416: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_414, mul_66);  mul_414 = None
        sum_135: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True);  mul_416 = None
        mul_417: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_135);  sum_135 = None
        sub_93: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_415, sum_134);  mul_415 = sum_134 = None
        sub_94: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_93, mul_417);  sub_93 = mul_417 = None
        mul_418: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_94);  div_30 = sub_94 = None
        mul_419: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_66);  mul_66 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 1]);  mul_419 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_420: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_421: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_418, mul_420);  mul_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_479: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_421, [16384, 768]);  mul_421 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_382: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_92: "f32[16384, 768]" = torch.ops.aten.mm.default(view_479, permute_382);  permute_382 = None
        permute_383: "f32[768, 16384]" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_383, view_104);  permute_383 = view_104 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
        view_480: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        view_481: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_92, [32, 512, 768]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_482: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_481, [32, 512, 12, 64]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_386: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_99: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_99, clone_default_21, clone_default_22, clone_default_23, None, getitem_101, getitem_102, getitem_103, getitem_104, 0.1, [True, True, True, False], scale = 0.125);  clone_99 = clone_default_21 = clone_default_22 = clone_default_23 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_105: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
        getitem_106: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_107: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_392: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_107, [0, 2, 1, 3]);  getitem_107 = None
        clone_101: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_489: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_101, [32, 512, 768]);  clone_101 = None
        view_490: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_489, [16384, 768]);  view_489 = None
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_94: "f32[16384, 768]" = torch.ops.aten.mm.default(view_490, permute_393);  permute_393 = None
        permute_394: "f32[768, 16384]" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
        sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
        view_491: "f32[768]" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
        view_492: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_94, [32, 512, 768]);  mm_94 = None
        add_157: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_418, view_492);  mul_418 = view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_397: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        view_493: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_397, [32, 512, 768]);  permute_397 = None
        clone_102: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_493, memory_format = torch.contiguous_format);  view_493 = None
        view_494: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_102, [16384, 768]);  clone_102 = None
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_96: "f32[16384, 768]" = torch.ops.aten.mm.default(view_494, permute_398);  permute_398 = None
        permute_399: "f32[768, 16384]" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_88);  permute_399 = None
        sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
        view_495: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        view_496: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_96, [32, 512, 768]);  mm_96 = None
        add_158: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_157, view_496);  add_157 = view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_402: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None
        clone_103: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
        view_497: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_103, [32, 512, 768]);  clone_103 = None
        view_498: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_497, [16384, 768]);  view_497 = None
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_403: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_98: "f32[16384, 768]" = torch.ops.aten.mm.default(view_498, permute_403);  permute_403 = None
        permute_404: "f32[768, 16384]" = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_404, view_88);  permute_404 = view_88 = None
        sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
        view_499: "f32[768]" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None
        view_500: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_98, [32, 512, 768]);  mm_98 = None
        add_159: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_158, view_500);  add_158 = view_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_427: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_74);  primals_74 = None
        mul_428: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, 768)
        sum_143: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
        mul_429: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, mul_59);  mul_427 = None
        sum_144: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
        mul_430: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, sum_144);  sum_144 = None
        sub_96: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_428, sum_143);  mul_428 = sum_143 = None
        sub_97: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_430);  sub_96 = mul_430 = None
        mul_431: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_97);  div_31 = sub_97 = None
        mul_432: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_59);  mul_59 = None
        sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_433: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_434: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_431, mul_433);  mul_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_434, [16384, 768]);  mul_434 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_407: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_100: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_501, permute_407);  permute_407 = None
        permute_408: "f32[768, 16384]" = torch.ops.aten.permute.default(view_501, [1, 0])
        mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_408, view_86);  permute_408 = view_86 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
        view_502: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_503: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_100, [32, 512, 3072]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_55: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_38: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_436: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
        mul_437: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_438: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_437, -0.5);  mul_437 = None
        exp_23: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_438);  mul_438 = None
        mul_439: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_440: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_439);  view_85 = mul_439 = None
        add_161: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
        mul_441: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, add_161);  view_503 = add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_441, [16384, 3072]);  mul_441 = None
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_411: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_102: "f32[16384, 768]" = torch.ops.aten.mm.default(view_504, permute_411);  permute_411 = None
        permute_412: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_412, view_84);  permute_412 = view_84 = None
        sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
        view_505: "f32[3072]" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
        view_506: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_102, [32, 512, 768]);  mm_102 = None
        add_162: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_431, view_506);  mul_431 = view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_443: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, primals_68);  primals_68 = None
        mul_444: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, 768)
        sum_149: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
        mul_445: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, mul_52);  mul_443 = None
        sum_150: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
        mul_446: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_150);  sum_150 = None
        sub_99: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_444, sum_149);  mul_444 = sum_149 = None
        sub_100: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_99, mul_446);  sub_99 = mul_446 = None
        mul_447: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_100);  div_32 = sub_100 = None
        mul_448: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, mul_52);  mul_52 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_449: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_450: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_447, mul_449);  mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_507: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_450, [16384, 768]);  mul_450 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_104: "f32[16384, 768]" = torch.ops.aten.mm.default(view_507, permute_415);  permute_415 = None
        permute_416: "f32[768, 16384]" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_82);  permute_416 = view_82 = None
        sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
        view_508: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        view_509: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_104, [32, 512, 768]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_510: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_509, [32, 512, 12, 64]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_419: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_106: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_106, clone_default_24, clone_default_25, clone_default_26, None, getitem_108, getitem_109, getitem_110, getitem_111, 0.1, [True, True, True, False], scale = 0.125);  clone_106 = clone_default_24 = clone_default_25 = clone_default_26 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_112: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
        getitem_113: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_114: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_425: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3]);  getitem_114 = None
        clone_108: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
        view_517: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_108, [32, 512, 768]);  clone_108 = None
        view_518: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_517, [16384, 768]);  view_517 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_426: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_106: "f32[16384, 768]" = torch.ops.aten.mm.default(view_518, permute_426);  permute_426 = None
        permute_427: "f32[768, 16384]" = torch.ops.aten.permute.default(view_518, [1, 0])
        mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
        sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
        view_519: "f32[768]" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
        view_520: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_106, [32, 512, 768]);  mm_106 = None
        add_163: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_447, view_520);  mul_447 = view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_430: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
        view_521: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_430, [32, 512, 768]);  permute_430 = None
        clone_109: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_521, memory_format = torch.contiguous_format);  view_521 = None
        view_522: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_109, [16384, 768]);  clone_109 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_108: "f32[16384, 768]" = torch.ops.aten.mm.default(view_522, permute_431);  permute_431 = None
        permute_432: "f32[768, 16384]" = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_66);  permute_432 = None
        sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
        view_523: "f32[768]" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
        view_524: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_108, [32, 512, 768]);  mm_108 = None
        add_164: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_524);  add_163 = view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_435: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None
        clone_110: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
        view_525: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_110, [32, 512, 768]);  clone_110 = None
        view_526: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_525, [16384, 768]);  view_525 = None
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_436: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_110: "f32[16384, 768]" = torch.ops.aten.mm.default(view_526, permute_436);  permute_436 = None
        permute_437: "f32[768, 16384]" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_437, view_66);  permute_437 = view_66 = None
        sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
        view_527: "f32[768]" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None
        view_528: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_110, [32, 512, 768]);  mm_110 = None
        add_165: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_164, view_528);  add_164 = view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_456: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_165, primals_58);  primals_58 = None
        mul_457: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_456, 768)
        sum_158: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True)
        mul_458: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_456, mul_45);  mul_456 = None
        sum_159: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [2], True);  mul_458 = None
        mul_459: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_159);  sum_159 = None
        sub_102: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_457, sum_158);  mul_457 = sum_158 = None
        sub_103: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_459);  sub_102 = mul_459 = None
        mul_460: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_103);  div_33 = sub_103 = None
        mul_461: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_165, mul_45);  mul_45 = None
        sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1]);  mul_461 = None
        sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_462: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_463: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_460, mul_462);  mul_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_463, [16384, 768]);  mul_463 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_112: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_529, permute_440);  permute_440 = None
        permute_441: "f32[768, 16384]" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_441, view_64);  permute_441 = view_64 = None
        sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        view_530: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        view_531: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_112, [32, 512, 3072]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_41: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_30: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_465: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
        mul_466: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_467: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_466, -0.5);  mul_466 = None
        exp_24: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_467);  mul_467 = None
        mul_468: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_469: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_468);  view_63 = mul_468 = None
        add_167: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_465, mul_469);  mul_465 = mul_469 = None
        mul_470: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_531, add_167);  view_531 = add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_532: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_470, [16384, 3072]);  mul_470 = None
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        permute_444: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_114: "f32[16384, 768]" = torch.ops.aten.mm.default(view_532, permute_444);  permute_444 = None
        permute_445: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_445, view_62);  permute_445 = view_62 = None
        sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
        view_533: "f32[3072]" = torch.ops.aten.reshape.default(sum_163, [3072]);  sum_163 = None
        view_534: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_114, [32, 512, 768]);  mm_114 = None
        add_168: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_460, view_534);  mul_460 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_472: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_168, primals_52);  primals_52 = None
        mul_473: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_472, 768)
        sum_164: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True)
        mul_474: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_472, mul_38);  mul_472 = None
        sum_165: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True);  mul_474 = None
        mul_475: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_165);  sum_165 = None
        sub_105: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_473, sum_164);  mul_473 = sum_164 = None
        sub_106: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_475);  sub_105 = mul_475 = None
        mul_476: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_106);  div_34 = sub_106 = None
        mul_477: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_168, mul_38);  mul_38 = None
        sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 1]);  mul_477 = None
        sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_478: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_479: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_476, mul_478);  mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_535: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_479, [16384, 768]);  mul_479 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_448: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_116: "f32[16384, 768]" = torch.ops.aten.mm.default(view_535, permute_448);  permute_448 = None
        permute_449: "f32[768, 16384]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_60);  permute_449 = view_60 = None
        sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[768]" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
        view_537: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_116, [32, 512, 768]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_538: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_537, [32, 512, 12, 64]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_452: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_113: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_113, clone_default_27, clone_default_28, clone_default_29, None, getitem_115, getitem_116, getitem_117, getitem_118, 0.1, [True, True, True, False], scale = 0.125);  clone_113 = clone_default_27 = clone_default_28 = clone_default_29 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_119: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
        getitem_120: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_121: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_458: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        clone_115: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_545: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_115, [32, 512, 768]);  clone_115 = None
        view_546: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_545, [16384, 768]);  view_545 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_118: "f32[16384, 768]" = torch.ops.aten.mm.default(view_546, permute_459);  permute_459 = None
        permute_460: "f32[768, 16384]" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
        sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
        view_547: "f32[768]" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
        view_548: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_118, [32, 512, 768]);  mm_118 = None
        add_169: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_476, view_548);  mul_476 = view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_463: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        view_549: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_463, [32, 512, 768]);  permute_463 = None
        clone_116: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_549, memory_format = torch.contiguous_format);  view_549 = None
        view_550: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_116, [16384, 768]);  clone_116 = None
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_464: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_120: "f32[16384, 768]" = torch.ops.aten.mm.default(view_550, permute_464);  permute_464 = None
        permute_465: "f32[768, 16384]" = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_465, view_44);  permute_465 = None
        sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
        view_551: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
        view_552: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_120, [32, 512, 768]);  mm_120 = None
        add_170: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_169, view_552);  add_169 = view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_468: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None
        clone_117: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_553: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_117, [32, 512, 768]);  clone_117 = None
        view_554: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_553, [16384, 768]);  view_553 = None
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_469: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_122: "f32[16384, 768]" = torch.ops.aten.mm.default(view_554, permute_469);  permute_469 = None
        permute_470: "f32[768, 16384]" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_470, view_44);  permute_470 = view_44 = None
        sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
        view_555: "f32[768]" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
        view_556: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_122, [32, 512, 768]);  mm_122 = None
        add_171: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_170, view_556);  add_170 = view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_485: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_171, primals_42);  primals_42 = None
        mul_486: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_485, 768)
        sum_173: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
        mul_487: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_485, mul_31);  mul_485 = None
        sum_174: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
        mul_488: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, sum_174);  sum_174 = None
        sub_108: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_486, sum_173);  mul_486 = sum_173 = None
        sub_109: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_488);  sub_108 = mul_488 = None
        mul_489: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_109);  div_35 = sub_109 = None
        mul_490: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_31);  mul_31 = None
        sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
        sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_491: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_492: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_489, mul_491);  mul_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_557: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_492, [16384, 768]);  mul_492 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_473: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_124: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_557, permute_473);  permute_473 = None
        permute_474: "f32[768, 16384]" = torch.ops.aten.permute.default(view_557, [1, 0])
        mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_474, view_42);  permute_474 = view_42 = None
        sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
        view_558: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        view_559: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_124, [32, 512, 3072]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_22: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_494: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
        mul_495: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_496: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_495, -0.5);  mul_495 = None
        exp_25: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_496);  mul_496 = None
        mul_497: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_498: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_497);  view_41 = mul_497 = None
        add_173: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_494, mul_498);  mul_494 = mul_498 = None
        mul_499: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_559, add_173);  view_559 = add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_560: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_499, [16384, 3072]);  mul_499 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_477: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_126: "f32[16384, 768]" = torch.ops.aten.mm.default(view_560, permute_477);  permute_477 = None
        permute_478: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_478, view_40);  permute_478 = view_40 = None
        sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
        view_561: "f32[3072]" = torch.ops.aten.reshape.default(sum_178, [3072]);  sum_178 = None
        view_562: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_126, [32, 512, 768]);  mm_126 = None
        add_174: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_489, view_562);  mul_489 = view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_501: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_174, primals_36);  primals_36 = None
        mul_502: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_501, 768)
        sum_179: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True)
        mul_503: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_501, mul_24);  mul_501 = None
        sum_180: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [2], True);  mul_503 = None
        mul_504: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_180);  sum_180 = None
        sub_111: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_502, sum_179);  mul_502 = sum_179 = None
        sub_112: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_504);  sub_111 = mul_504 = None
        mul_505: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_112);  div_36 = sub_112 = None
        mul_506: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_174, mul_24);  mul_24 = None
        sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 1]);  mul_506 = None
        sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_174, [0, 1]);  add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_507: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_508: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_505, mul_507);  mul_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_563: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_508, [16384, 768]);  mul_508 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_128: "f32[16384, 768]" = torch.ops.aten.mm.default(view_563, permute_481);  permute_481 = None
        permute_482: "f32[768, 16384]" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_482, view_38);  permute_482 = view_38 = None
        sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        view_564: "f32[768]" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
        view_565: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_128, [32, 512, 768]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_566: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_565, [32, 512, 12, 64]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_485: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_120: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_120, clone_default_30, clone_default_31, clone_default_32, None, getitem_122, getitem_123, getitem_124, getitem_125, 0.1, [True, True, True, False], scale = 0.125);  clone_120 = clone_default_30 = clone_default_31 = clone_default_32 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_126: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
        getitem_127: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_128: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_491: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None
        clone_122: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_573: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_122, [32, 512, 768]);  clone_122 = None
        view_574: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_573, [16384, 768]);  view_573 = None
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_130: "f32[16384, 768]" = torch.ops.aten.mm.default(view_574, permute_492);  permute_492 = None
        permute_493: "f32[768, 16384]" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
        sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
        view_575: "f32[768]" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
        view_576: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_130, [32, 512, 768]);  mm_130 = None
        add_175: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_505, view_576);  mul_505 = view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_496: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None
        view_577: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_496, [32, 512, 768]);  permute_496 = None
        clone_123: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_577, memory_format = torch.contiguous_format);  view_577 = None
        view_578: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_123, [16384, 768]);  clone_123 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_132: "f32[16384, 768]" = torch.ops.aten.mm.default(view_578, permute_497);  permute_497 = None
        permute_498: "f32[768, 16384]" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_498, view_22);  permute_498 = None
        sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True);  view_578 = None
        view_579: "f32[768]" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
        view_580: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_132, [32, 512, 768]);  mm_132 = None
        add_176: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_580);  add_175 = view_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_501: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        clone_124: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_581: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_124, [32, 512, 768]);  clone_124 = None
        view_582: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_581, [16384, 768]);  view_581 = None
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_502: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_134: "f32[16384, 768]" = torch.ops.aten.mm.default(view_582, permute_502);  permute_502 = None
        permute_503: "f32[768, 16384]" = torch.ops.aten.permute.default(view_582, [1, 0])
        mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_503, view_22);  permute_503 = view_22 = None
        sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
        view_583: "f32[768]" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None
        view_584: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_134, [32, 512, 768]);  mm_134 = None
        add_177: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_176, view_584);  add_176 = view_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_514: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, primals_26);  primals_26 = None
        mul_515: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_514, 768)
        sum_188: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True)
        mul_516: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_514, mul_17);  mul_514 = None
        sum_189: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True);  mul_516 = None
        mul_517: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_189);  sum_189 = None
        sub_114: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_515, sum_188);  mul_515 = sum_188 = None
        sub_115: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_517);  sub_114 = mul_517 = None
        mul_518: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_115);  div_37 = sub_115 = None
        mul_519: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, mul_17);  mul_17 = None
        sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 1]);  mul_519 = None
        sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_177, [0, 1]);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_520: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_521: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_518, mul_520);  mul_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_585: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_521, [16384, 768]);  mul_521 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_506: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_136: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_585, permute_506);  permute_506 = None
        permute_507: "f32[768, 16384]" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_507, view_20);  permute_507 = view_20 = None
        sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
        view_586: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        view_587: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_136, [32, 512, 3072]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_14: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_523: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_524: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_525: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_524, -0.5);  mul_524 = None
        exp_26: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_525);  mul_525 = None
        mul_526: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_527: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_526);  view_19 = mul_526 = None
        add_179: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_523, mul_527);  mul_523 = mul_527 = None
        mul_528: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_587, add_179);  view_587 = add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_588: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_528, [16384, 3072]);  mul_528 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_510: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_138: "f32[16384, 768]" = torch.ops.aten.mm.default(view_588, permute_510);  permute_510 = None
        permute_511: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_588, [1, 0])
        mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_511, view_18);  permute_511 = view_18 = None
        sum_193: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
        view_589: "f32[3072]" = torch.ops.aten.reshape.default(sum_193, [3072]);  sum_193 = None
        view_590: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_138, [32, 512, 768]);  mm_138 = None
        add_180: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_518, view_590);  mul_518 = view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_530: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_180, primals_20);  primals_20 = None
        mul_531: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_530, 768)
        sum_194: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True)
        mul_532: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_530, mul_10);  mul_530 = None
        sum_195: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True);  mul_532 = None
        mul_533: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_195);  sum_195 = None
        sub_117: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_531, sum_194);  mul_531 = sum_194 = None
        sub_118: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_533);  sub_117 = mul_533 = None
        mul_534: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_118);  div_38 = sub_118 = None
        mul_535: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_180, mul_10);  mul_10 = None
        sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 1]);  mul_535 = None
        sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_180, [0, 1]);  add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_536: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_537: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_534, mul_536);  mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_591: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_537, [16384, 768]);  mul_537 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_514: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_140: "f32[16384, 768]" = torch.ops.aten.mm.default(view_591, permute_514);  permute_514 = None
        permute_515: "f32[768, 16384]" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_16);  permute_515 = view_16 = None
        sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
        view_592: "f32[768]" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
        view_593: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_140, [32, 512, 768]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_594: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_593, [32, 512, 12, 64]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_518: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_127: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_127, clone_default_33, clone_default_34, clone_default_35, None, getitem_129, getitem_130, getitem_131, getitem_132, 0.1, [True, True, True, False], scale = 0.125);  clone_127 = clone_default_33 = clone_default_34 = clone_default_35 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        getitem_133: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
        getitem_134: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_135: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_524: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
        clone_129: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        view_601: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_129, [32, 512, 768]);  clone_129 = None
        view_602: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_601, [16384, 768]);  view_601 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_525: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_142: "f32[16384, 768]" = torch.ops.aten.mm.default(view_602, permute_525);  permute_525 = None
        permute_526: "f32[768, 16384]" = torch.ops.aten.permute.default(view_602, [1, 0])
        mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
        sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
        view_603: "f32[768]" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
        view_604: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [32, 512, 768]);  mm_142 = None
        add_181: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_534, view_604);  mul_534 = view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_529: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_605: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_529, [32, 512, 768]);  permute_529 = None
        clone_130: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_605, memory_format = torch.contiguous_format);  view_605 = None
        view_606: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_130, [16384, 768]);  clone_130 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_144: "f32[16384, 768]" = torch.ops.aten.mm.default(view_606, permute_530);  permute_530 = None
        permute_531: "f32[768, 16384]" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_531, view);  permute_531 = None
        sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
        view_607: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
        view_608: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [32, 512, 768]);  mm_144 = None
        add_182: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_181, view_608);  add_181 = view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_534: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        clone_131: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
        view_609: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_131, [32, 512, 768]);  clone_131 = None
        view_610: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_609, [16384, 768]);  view_609 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_535: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_146: "f32[16384, 768]" = torch.ops.aten.mm.default(view_610, permute_535);  permute_535 = None
        permute_536: "f32[768, 16384]" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_536, view);  permute_536 = view = None
        sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_610, [0], True);  view_610 = None
        view_611: "f32[768]" = torch.ops.aten.reshape.default(sum_202, [768]);  sum_202 = None
        view_612: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [32, 512, 768]);  mm_146 = None
        add_183: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_182, view_612);  add_182 = view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:120 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_37: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_542: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_543: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_183, mul_542);  add_183 = mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_545: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_543, primals_10);  primals_10 = None
        mul_546: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_545, 768)
        sum_203: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True)
        mul_547: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_545, mul_1);  mul_545 = None
        sum_204: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
        mul_548: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_204);  sum_204 = None
        sub_120: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_546, sum_203);  mul_546 = sum_203 = None
        sub_121: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_548);  sub_120 = mul_548 = None
        mul_549: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_121);  div_39 = sub_121 = None
        mul_550: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_543, mul_1);  mul_1 = None
        sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1]);  mul_550 = None
        sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1]);  mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        sum_207: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_549, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_8: "b8[32, 512, 1]" = torch.ops.aten.full.default([32, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[32, 512, 768]" = torch.ops.aten.where.self(full_default_8, full_default_4, mul_549);  full_default_8 = None
        full_default_10: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_10, [full_default], where_4, True);  full_default_10 = full_default = where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        eq_1: "b8[32, 512]" = torch.ops.aten.eq.Scalar(sub_2, -1)
        unsqueeze_5: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_5: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_5, full_default_4, mul_549);  unsqueeze_5 = None
        full_default_12: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_12, [sub_2], where_5, True);  sub_2 = where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        eq_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(sub_1, -1)
        unsqueeze_6: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_6: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_6, full_default_4, mul_549);  unsqueeze_6 = None
        index_put_2: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_12, [sub_1], where_6, True);  sub_1 = where_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:100 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        eq_3: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select_3, -1)
        unsqueeze_7: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
        where_7: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_7, full_default_4, mul_549);  unsqueeze_7 = None
        index_put_3: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_12, [select_3], where_7, True);  select_3 = where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        eq_4: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select_2, -1)
        unsqueeze_8: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_4, -1);  eq_4 = None
        where_8: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_4, mul_549);  unsqueeze_8 = None
        index_put_4: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_12, [select_2], where_8, True);  select_2 = where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:98 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        eq_5: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select_1, -1)
        unsqueeze_9: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_5, -1);  eq_5 = None
        where_9: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_9, full_default_4, mul_549);  unsqueeze_9 = None
        index_put_5: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_12, [select_1], where_9, True);  select_1 = where_9 = None
        add_184: "f32[1024, 768]" = torch.ops.aten.add.Tensor(index_put_3, index_put_5);  index_put_3 = index_put_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        eq_6: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select, -1)
        unsqueeze_10: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_6, -1);  eq_6 = None
        where_10: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_10, full_default_4, mul_549);  unsqueeze_10 = None
        index_put_6: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_12, [select], where_10, True);  full_default_12 = select = where_10 = None
        add_185: "f32[1024, 768]" = torch.ops.aten.add.Tensor(index_put_4, index_put_6);  index_put_4 = index_put_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_7: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_11: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_7, -1);  eq_7 = None
        where_11: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_11, full_default_4, sum_207);  unsqueeze_11 = sum_207 = None
        full_default_24: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_7: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_24, [primals_3], where_11, True);  full_default_24 = primals_3 = where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_8: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_12: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_8, -1);  eq_8 = None
        where_12: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_12, full_default_4, mul_549);  unsqueeze_12 = full_default_4 = mul_549 = None
        full_default_26: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_8: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_26, [primals_1], where_12, True);  full_default_26 = primals_1 = where_12 = None
        add_186: "f32[30522, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_8);  slice_tensor = index_put_8 = None
        return (None, add_186, None, index_put_7, add_185, add_184, index_put_2, index_put_1, index_put, sum_205, sum_206, mm_147, view_611, mm_145, view_607, mm_143, view_603, mm_141, view_592, sum_196, sum_197, mm_139, view_589, mm_137, view_586, sum_190, sum_191, mm_135, view_583, mm_133, view_579, mm_131, view_575, mm_129, view_564, sum_181, sum_182, mm_127, view_561, mm_125, view_558, sum_175, sum_176, mm_123, view_555, mm_121, view_551, mm_119, view_547, mm_117, view_536, sum_166, sum_167, mm_115, view_533, mm_113, view_530, sum_160, sum_161, mm_111, view_527, mm_109, view_523, mm_107, view_519, mm_105, view_508, sum_151, sum_152, mm_103, view_505, mm_101, view_502, sum_145, sum_146, mm_99, view_499, mm_97, view_495, mm_95, view_491, mm_93, view_480, sum_136, sum_137, mm_91, view_477, mm_89, view_474, sum_130, sum_131, mm_87, view_471, mm_85, view_467, mm_83, view_463, mm_81, view_452, sum_121, sum_122, mm_79, view_449, mm_77, view_446, sum_115, sum_116, mm_75, view_443, mm_73, view_439, mm_71, view_435, mm_69, view_424, sum_106, sum_107, mm_67, view_421, mm_65, view_418, sum_100, sum_101, mm_63, view_415, mm_61, view_411, mm_59, view_407, mm_57, view_396, sum_91, sum_92, mm_55, view_393, mm_53, view_390, sum_85, sum_86, mm_51, view_387, mm_49, view_383, mm_47, view_379, mm_45, view_368, sum_76, sum_77, mm_43, view_365, mm_41, view_362, sum_70, sum_71, mm_39, view_359, mm_37, view_355, mm_35, view_351, mm_33, view_340, sum_61, sum_62, mm_31, view_337, mm_29, view_334, sum_55, sum_56, mm_27, view_331, mm_25, view_327, mm_23, view_323, mm_21, view_312, sum_46, sum_47, mm_19, view_309, mm_17, view_306, sum_40, sum_41, mm_15, view_303, mm_13, view_299, mm_11, view_295, mm_9, view_284, sum_31, sum_32, mm_7, view_281, mm_5, view_278, sum_25, sum_26, None, None, mm_3, view_275, sum_20, sum_21, view_272, None)
