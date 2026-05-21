class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[128, 128]", primals_5: "i64[128, 128]", primals_6: "f32[768]", primals_8: "f32[768, 768]", primals_10: "f32[768, 768]", primals_12: "f32[768, 768]", primals_14: "f32[768, 768]", primals_16: "f32[768]", primals_18: "f32[3072, 768]", primals_20: "f32[768, 3072]", primals_22: "f32[768]", primals_24: "f32[768, 768]", primals_26: "f32[768, 768]", primals_28: "f32[768, 768]", primals_30: "f32[768, 768]", primals_32: "f32[768]", primals_34: "f32[3072, 768]", primals_36: "f32[768, 3072]", primals_38: "f32[768]", primals_40: "f32[768, 768]", primals_42: "f32[768, 768]", primals_44: "f32[768, 768]", primals_46: "f32[768, 768]", primals_48: "f32[768]", primals_50: "f32[3072, 768]", primals_52: "f32[768, 3072]", primals_54: "f32[768]", primals_56: "f32[768, 768]", primals_58: "f32[768, 768]", primals_60: "f32[768, 768]", primals_62: "f32[768, 768]", primals_64: "f32[768]", primals_66: "f32[3072, 768]", primals_68: "f32[768, 3072]", primals_70: "f32[768]", primals_72: "f32[768, 768]", primals_74: "f32[768, 768]", primals_76: "f32[768, 768]", primals_78: "f32[768, 768]", primals_80: "f32[768]", primals_82: "f32[3072, 768]", primals_84: "f32[768, 3072]", primals_86: "f32[768]", primals_88: "f32[768, 768]", primals_90: "f32[768, 768]", primals_92: "f32[768, 768]", primals_94: "f32[768, 768]", primals_96: "f32[768]", primals_98: "f32[3072, 768]", primals_100: "f32[768, 3072]", primals_102: "f32[768]", primals_104: "f32[768, 768]", primals_106: "f32[768, 768]", primals_108: "f32[768, 768]", primals_110: "f32[768, 768]", primals_112: "f32[768]", primals_114: "f32[3072, 768]", primals_116: "f32[768, 3072]", primals_118: "f32[768]", primals_120: "f32[768, 768]", primals_122: "f32[768, 768]", primals_124: "f32[768, 768]", primals_126: "f32[768, 768]", primals_128: "f32[768]", primals_130: "f32[3072, 768]", primals_132: "f32[768, 3072]", primals_134: "f32[768]", primals_136: "f32[768, 768]", primals_138: "f32[768, 768]", primals_140: "f32[768, 768]", primals_142: "f32[768, 768]", primals_144: "f32[768]", primals_146: "f32[3072, 768]", primals_148: "f32[768, 3072]", primals_150: "f32[768]", primals_152: "f32[768, 768]", primals_154: "f32[768, 768]", primals_156: "f32[768, 768]", primals_158: "f32[768, 768]", primals_160: "f32[768]", primals_162: "f32[3072, 768]", primals_164: "f32[768, 3072]", primals_166: "f32[768]", primals_168: "f32[768, 768]", primals_170: "f32[768, 768]", primals_172: "f32[768, 768]", primals_174: "f32[768, 768]", primals_176: "f32[768]", primals_178: "f32[3072, 768]", primals_180: "f32[768, 3072]", primals_182: "f32[768]", primals_184: "f32[768, 768]", primals_186: "f32[768, 768]", primals_188: "f32[768, 768]", primals_190: "f32[768, 768]", primals_192: "f32[768]", primals_194: "f32[3072, 768]", primals_196: "f32[768, 3072]", primals_198: "f32[2, 768]", primals_200: "f32[20005, 768]", unsqueeze_1: "b8[128, 1, 128, 128]", gt_1: "b8[128, 128, 768]", sqrt: "f32[128, 128, 1]", sub: "f32[128, 128, 768]", view: "f32[16384, 768]", bmm: "f32[1536, 128, 128]", amax: "f32[128, 12, 128, 1]", sum_1: "f32[128, 12, 128, 1]", gt_2: "b8[128, 12, 128, 128]", view_16: "f32[16384, 768]", gt_3: "b8[128, 128, 768]", sqrt_1: "f32[128, 128, 1]", sub_2: "f32[128, 128, 768]", view_18: "f32[16384, 768]", addmm_4: "f32[16384, 3072]", gt_4: "b8[128, 128, 3072]", view_20: "f32[16384, 3072]", gt_5: "b8[128, 128, 768]", gt_6: "b8[128, 128, 768]", sqrt_2: "f32[128, 128, 1]", sub_3: "f32[128, 128, 768]", view_22: "f32[16384, 768]", div_6: "f32[128, 12, 128, 128]", gt_7: "b8[128, 12, 128, 128]", view_38: "f32[16384, 768]", gt_8: "b8[128, 128, 768]", sqrt_3: "f32[128, 128, 1]", sub_5: "f32[128, 128, 768]", view_40: "f32[16384, 768]", addmm_10: "f32[16384, 3072]", gt_9: "b8[128, 128, 3072]", view_42: "f32[16384, 3072]", gt_10: "b8[128, 128, 768]", gt_11: "b8[128, 128, 768]", sqrt_4: "f32[128, 128, 1]", sub_6: "f32[128, 128, 768]", view_44: "f32[16384, 768]", div_10: "f32[128, 12, 128, 128]", gt_12: "b8[128, 12, 128, 128]", view_60: "f32[16384, 768]", gt_13: "b8[128, 128, 768]", sqrt_5: "f32[128, 128, 1]", sub_8: "f32[128, 128, 768]", view_62: "f32[16384, 768]", addmm_16: "f32[16384, 3072]", gt_14: "b8[128, 128, 3072]", view_64: "f32[16384, 3072]", gt_15: "b8[128, 128, 768]", gt_16: "b8[128, 128, 768]", sqrt_6: "f32[128, 128, 1]", sub_9: "f32[128, 128, 768]", view_66: "f32[16384, 768]", div_14: "f32[128, 12, 128, 128]", gt_17: "b8[128, 12, 128, 128]", view_82: "f32[16384, 768]", gt_18: "b8[128, 128, 768]", sqrt_7: "f32[128, 128, 1]", sub_11: "f32[128, 128, 768]", view_84: "f32[16384, 768]", addmm_22: "f32[16384, 3072]", gt_19: "b8[128, 128, 3072]", view_86: "f32[16384, 3072]", gt_20: "b8[128, 128, 768]", gt_21: "b8[128, 128, 768]", sqrt_8: "f32[128, 128, 1]", sub_12: "f32[128, 128, 768]", view_88: "f32[16384, 768]", div_18: "f32[128, 12, 128, 128]", gt_22: "b8[128, 12, 128, 128]", view_104: "f32[16384, 768]", gt_23: "b8[128, 128, 768]", sqrt_9: "f32[128, 128, 1]", sub_14: "f32[128, 128, 768]", view_106: "f32[16384, 768]", addmm_28: "f32[16384, 3072]", gt_24: "b8[128, 128, 3072]", view_108: "f32[16384, 3072]", gt_25: "b8[128, 128, 768]", gt_26: "b8[128, 128, 768]", sqrt_10: "f32[128, 128, 1]", sub_15: "f32[128, 128, 768]", view_110: "f32[16384, 768]", div_22: "f32[128, 12, 128, 128]", gt_27: "b8[128, 12, 128, 128]", view_126: "f32[16384, 768]", gt_28: "b8[128, 128, 768]", sqrt_11: "f32[128, 128, 1]", sub_17: "f32[128, 128, 768]", view_128: "f32[16384, 768]", addmm_34: "f32[16384, 3072]", gt_29: "b8[128, 128, 3072]", view_130: "f32[16384, 3072]", gt_30: "b8[128, 128, 768]", gt_31: "b8[128, 128, 768]", sqrt_12: "f32[128, 128, 1]", sub_18: "f32[128, 128, 768]", view_132: "f32[16384, 768]", div_26: "f32[128, 12, 128, 128]", gt_32: "b8[128, 12, 128, 128]", view_148: "f32[16384, 768]", gt_33: "b8[128, 128, 768]", sqrt_13: "f32[128, 128, 1]", sub_20: "f32[128, 128, 768]", view_150: "f32[16384, 768]", addmm_40: "f32[16384, 3072]", gt_34: "b8[128, 128, 3072]", view_152: "f32[16384, 3072]", gt_35: "b8[128, 128, 768]", gt_36: "b8[128, 128, 768]", sqrt_14: "f32[128, 128, 1]", sub_21: "f32[128, 128, 768]", view_154: "f32[16384, 768]", div_30: "f32[128, 12, 128, 128]", gt_37: "b8[128, 12, 128, 128]", view_170: "f32[16384, 768]", gt_38: "b8[128, 128, 768]", sqrt_15: "f32[128, 128, 1]", sub_23: "f32[128, 128, 768]", view_172: "f32[16384, 768]", addmm_46: "f32[16384, 3072]", gt_39: "b8[128, 128, 3072]", view_174: "f32[16384, 3072]", gt_40: "b8[128, 128, 768]", gt_41: "b8[128, 128, 768]", sqrt_16: "f32[128, 128, 1]", sub_24: "f32[128, 128, 768]", view_176: "f32[16384, 768]", div_34: "f32[128, 12, 128, 128]", gt_42: "b8[128, 12, 128, 128]", view_192: "f32[16384, 768]", gt_43: "b8[128, 128, 768]", sqrt_17: "f32[128, 128, 1]", sub_26: "f32[128, 128, 768]", view_194: "f32[16384, 768]", addmm_52: "f32[16384, 3072]", gt_44: "b8[128, 128, 3072]", view_196: "f32[16384, 3072]", gt_45: "b8[128, 128, 768]", gt_46: "b8[128, 128, 768]", sqrt_18: "f32[128, 128, 1]", sub_27: "f32[128, 128, 768]", view_198: "f32[16384, 768]", div_38: "f32[128, 12, 128, 128]", gt_47: "b8[128, 12, 128, 128]", view_214: "f32[16384, 768]", gt_48: "b8[128, 128, 768]", sqrt_19: "f32[128, 128, 1]", sub_29: "f32[128, 128, 768]", view_216: "f32[16384, 768]", addmm_58: "f32[16384, 3072]", gt_49: "b8[128, 128, 3072]", view_218: "f32[16384, 3072]", gt_50: "b8[128, 128, 768]", gt_51: "b8[128, 128, 768]", sqrt_20: "f32[128, 128, 1]", sub_30: "f32[128, 128, 768]", view_220: "f32[16384, 768]", div_42: "f32[128, 12, 128, 128]", gt_52: "b8[128, 12, 128, 128]", view_236: "f32[16384, 768]", gt_53: "b8[128, 128, 768]", sqrt_21: "f32[128, 128, 1]", sub_32: "f32[128, 128, 768]", view_238: "f32[16384, 768]", addmm_64: "f32[16384, 3072]", gt_54: "b8[128, 128, 3072]", view_240: "f32[16384, 3072]", gt_55: "b8[128, 128, 768]", gt_56: "b8[128, 128, 768]", sqrt_22: "f32[128, 128, 1]", sub_33: "f32[128, 128, 768]", view_242: "f32[16384, 768]", div_46: "f32[128, 12, 128, 128]", gt_57: "b8[128, 12, 128, 128]", view_258: "f32[16384, 768]", gt_58: "b8[128, 128, 768]", sqrt_23: "f32[128, 128, 1]", sub_35: "f32[128, 128, 768]", view_260: "f32[16384, 768]", addmm_70: "f32[16384, 3072]", gt_59: "b8[128, 128, 3072]", view_262: "f32[16384, 3072]", gt_60: "b8[128, 128, 768]", gt_61: "b8[128, 128, 768]", select: "f32[128, 768]", sub_37: "f32[128, 2]", view_264: "f32[16384, 768]", sub_39: "f32[128, 128, 20005]", permute_155: "f32[1536, 128, 128]", permute_156: "f32[1536, 64, 128]", permute_157: "f32[1536, 64, 128]", permute_158: "f32[1536, 128, 64]", permute_188: "f32[1536, 128, 128]", permute_189: "f32[1536, 64, 128]", permute_190: "f32[1536, 64, 128]", permute_191: "f32[1536, 128, 64]", permute_221: "f32[1536, 128, 128]", permute_222: "f32[1536, 64, 128]", permute_223: "f32[1536, 64, 128]", permute_224: "f32[1536, 128, 64]", permute_254: "f32[1536, 128, 128]", permute_255: "f32[1536, 64, 128]", permute_256: "f32[1536, 64, 128]", permute_257: "f32[1536, 128, 64]", permute_287: "f32[1536, 128, 128]", permute_288: "f32[1536, 64, 128]", permute_289: "f32[1536, 64, 128]", permute_290: "f32[1536, 128, 64]", permute_320: "f32[1536, 128, 128]", permute_321: "f32[1536, 64, 128]", permute_322: "f32[1536, 64, 128]", permute_323: "f32[1536, 128, 64]", permute_353: "f32[1536, 128, 128]", permute_354: "f32[1536, 64, 128]", permute_355: "f32[1536, 64, 128]", permute_356: "f32[1536, 128, 64]", permute_386: "f32[1536, 128, 128]", permute_387: "f32[1536, 64, 128]", permute_388: "f32[1536, 64, 128]", permute_389: "f32[1536, 128, 64]", permute_419: "f32[1536, 128, 128]", permute_420: "f32[1536, 64, 128]", permute_421: "f32[1536, 64, 128]", permute_422: "f32[1536, 128, 64]", permute_452: "f32[1536, 128, 128]", permute_453: "f32[1536, 64, 128]", permute_454: "f32[1536, 64, 128]", permute_455: "f32[1536, 128, 64]", permute_485: "f32[1536, 128, 128]", permute_486: "f32[1536, 64, 128]", permute_487: "f32[1536, 64, 128]", permute_488: "f32[1536, 128, 64]", permute_518: "f32[1536, 128, 128]", permute_519: "f32[1536, 64, 128]", permute_520: "f32[1536, 64, 128]", permute_521: "f32[1536, 128, 64]", tangents_1: "f32[128, 2]", tangents_2: "f32[128, 128, 20005]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        sum_15: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(tangents_2, [-1], True)
        exp_14: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        mul_182: "f32[128, 128, 20005]" = torch.ops.aten.mul.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        sub_40: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(tangents_2, mul_182);  tangents_2 = mul_182 = None
        view_266: "f32[16384, 20005]" = torch.ops.aten.reshape.default(sub_40, [16384, 20005]);  sub_40 = None
        permute_133: "f32[768, 20005]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        permute_134: "f32[20005, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(view_266, permute_134);  permute_134 = None
        permute_135: "f32[20005, 16384]" = torch.ops.aten.permute.default(view_266, [1, 0])
        mm_1: "f32[20005, 768]" = torch.ops.aten.mm.default(permute_135, view_264);  permute_135 = view_264 = None
        sum_16: "f32[1, 20005]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
        view_267: "f32[20005]" = torch.ops.aten.reshape.default(sum_16, [20005]);  sum_16 = None
        view_268: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm, [128, 128, 768]);  mm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        exp_15: "f32[128, 2]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_17: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(tangents_1, [-1], True)
        mul_183: "f32[128, 2]" = torch.ops.aten.mul.Tensor(exp_15, sum_17);  exp_15 = sum_17 = None
        sub_41: "f32[128, 2]" = torch.ops.aten.sub.Tensor(tangents_1, mul_183);  tangents_1 = mul_183 = None
        permute_132: "f32[768, 2]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_138: "f32[2, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_2: "f32[128, 768]" = torch.ops.aten.mm.default(sub_41, permute_138);  permute_138 = None
        permute_139: "f32[2, 128]" = torch.ops.aten.permute.default(sub_41, [1, 0])
        mm_3: "f32[2, 768]" = torch.ops.aten.mm.default(permute_139, select);  permute_139 = select = None
        sum_18: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(sub_41, [0], True);  sub_41 = None
        view_269: "f32[2]" = torch.ops.aten.reshape.default(sum_18, [2]);  sum_18 = None
        full_default_12: "f32[128, 128, 768]" = torch.ops.aten.full.default([128, 128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[128, 128, 768]" = torch.ops.aten.select_scatter.default(full_default_12, mm_2, 1, 0);  full_default_12 = mm_2 = None
        add_86: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_268, select_scatter);  view_268 = select_scatter = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_184: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_185: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_86, mul_184);  add_86 = mul_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_186: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_187: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_186);  mul_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_270: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_187, [16384, 768]);  mul_187 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_4: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_270, permute_142);  permute_142 = None
        permute_143: "f32[768, 16384]" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
        sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
        view_271: "f32[768]" = torch.ops.aten.reshape.default(sum_19, [768]);  sum_19 = None
        view_272: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_4, [128, 128, 3072]);  mm_4 = None
        convert_element_type_2: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_188: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_189: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_272, mul_188);  view_272 = mul_188 = None
        view_261: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_70, [128, 128, 3072]);  addmm_70 = None
        mul_174: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_84: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_191: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
        mul_192: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_193: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_192, -0.5);  mul_192 = None
        exp_16: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_193);  mul_193 = None
        mul_194: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_195: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_194);  view_261 = mul_194 = None
        add_88: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_191, mul_195);  mul_191 = mul_195 = None
        mul_196: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_189, add_88);  mul_189 = add_88 = None
        view_273: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_196, [16384, 3072]);  mul_196 = None
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_6: "f32[16384, 768]" = torch.ops.aten.mm.default(view_273, permute_146);  permute_146 = None
        permute_147: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
        sum_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        view_274: "f32[3072]" = torch.ops.aten.reshape.default(sum_20, [3072]);  sum_20 = None
        view_275: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_6, [128, 128, 768]);  mm_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_21: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1], True)
        view_276: "f32[768]" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
        mul_172: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_192, sub_35)
        add_82: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06)
        div_47: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_172, add_82);  mul_172 = None
        div_49: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_47, add_82);  div_47 = None
        neg: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_275)
        mul_197: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_49);  neg = div_49 = None
        div_50: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_275, add_82);  view_275 = add_82 = None
        sum_22: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, primals_192);  primals_192 = None
        mul_199: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_35);  div_50 = None
        sum_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1], True);  mul_199 = None
        view_277: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
        neg_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_198)
        sum_24: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_1, [2], True);  neg_1 = None
        add_89: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_185, mul_198);  mul_185 = mul_198 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_200: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_23, 2)
        div_51: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_22, mul_200);  sum_22 = mul_200 = None
        eq_12: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_23, 0);  sqrt_23 = None
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_12, full_default_13, div_51);  eq_12 = div_51 = None
        mul_201: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_12, 0.002607561929595828);  where_12 = None
        mul_202: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_201, sub_35);  mul_201 = sub_35 = None
        add_90: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_89, mul_202);  add_89 = mul_202 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_48: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_24, [128, 128, 768]);  sum_24 = None
        div_52: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_48, 768);  expand_48 = None
        add_91: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_90, div_52);  add_90 = div_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_3: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_203: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_204: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, mul_203);  mul_203 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_278: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_204, [16384, 768]);  mul_204 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_8: "f32[16384, 768]" = torch.ops.aten.mm.default(view_278, permute_150);  permute_150 = None
        permute_151: "f32[768, 16384]" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
        sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        view_279: "f32[768]" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
        view_280: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_8, [128, 128, 768]);  mm_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_281: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_280, [128, 128, 12, 64]);  view_280 = None
        permute_154: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_52: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_282: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_52, [1536, 128, 64]);  clone_52 = None
        bmm_24: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_155, view_282);  permute_155 = None
        bmm_25: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_282, permute_156);  view_282 = permute_156 = None
        view_283: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_24, [128, 12, 128, 64]);  bmm_24 = None
        view_284: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, [128, 12, 128, 128]);  bmm_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_4: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_205: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_206: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_284, mul_205);  view_284 = mul_205 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_207: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_206, div_46);  mul_206 = None
        sum_26: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [-1], True)
        neg_2: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_26, mul_207);  neg_2 = sum_26 = mul_207 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        eq: "b8[128, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0);  unsqueeze_1 = None
        where_13: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma);  fma = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_53: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_13, 8.0);  where_13 = None
        view_285: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_53, [1536, 128, 128]);  div_53 = None
        bmm_26: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_157, view_285);  permute_157 = None
        bmm_27: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_285, permute_158);  view_285 = permute_158 = None
        view_286: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_26, [128, 12, 64, 128]);  bmm_26 = None
        view_287: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_27, [128, 12, 128, 64]);  bmm_27 = None
        permute_159: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_286, [0, 1, 3, 2]);  view_286 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_160: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        clone_54: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_288: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_54, [128, 128, 768]);  clone_54 = None
        view_289: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_288, [16384, 768]);  view_288 = None
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_10: "f32[16384, 768]" = torch.ops.aten.mm.default(view_289, permute_161);  permute_161 = None
        permute_162: "f32[768, 16384]" = torch.ops.aten.permute.default(view_289, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_242);  permute_162 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
        view_290: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_291: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_10, [128, 128, 768]);  mm_10 = None
        permute_165: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        view_292: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_165, [128, 128, 768]);  permute_165 = None
        clone_55: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_292, memory_format = torch.contiguous_format);  view_292 = None
        view_293: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_55, [16384, 768]);  clone_55 = None
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_12: "f32[16384, 768]" = torch.ops.aten.mm.default(view_293, permute_166);  permute_166 = None
        permute_167: "f32[768, 16384]" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = None
        sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        view_294: "f32[768]" = torch.ops.aten.reshape.default(sum_28, [768]);  sum_28 = None
        view_295: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_12, [128, 128, 768]);  mm_12 = None
        add_92: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_291, view_295);  view_291 = view_295 = None
        permute_170: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        clone_56: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_296: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_56, [128, 128, 768]);  clone_56 = None
        view_297: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_296, [16384, 768]);  view_296 = None
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_14: "f32[16384, 768]" = torch.ops.aten.mm.default(view_297, permute_171);  permute_171 = None
        permute_172: "f32[768, 16384]" = torch.ops.aten.permute.default(view_297, [1, 0])
        mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
        sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
        view_298: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
        view_299: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_14, [128, 128, 768]);  mm_14 = None
        add_93: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_92, view_299);  add_92 = view_299 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_93, [0, 1], True)
        view_300: "f32[768]" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
        mul_167: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_182, sub_33)
        add_79: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_22, 1e-06)
        div_44: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_167, add_79);  mul_167 = None
        div_55: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_44, add_79);  div_44 = None
        neg_3: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_93)
        mul_208: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_3, div_55);  neg_3 = div_55 = None
        div_56: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_93, add_79);  add_93 = add_79 = None
        sum_31: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_56, primals_182);  primals_182 = None
        mul_210: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_33);  div_56 = None
        sum_32: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True);  mul_210 = None
        view_301: "f32[768]" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
        neg_4: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_209)
        sum_33: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_4, [2], True);  neg_4 = None
        add_94: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_91, mul_209);  add_91 = mul_209 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_211: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_22, 2)
        div_57: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_31, mul_211);  sum_31 = mul_211 = None
        eq_13: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_22, 0);  sqrt_22 = None
        where_14: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_13, full_default_13, div_57);  eq_13 = div_57 = None
        mul_212: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_14, 0.002607561929595828);  where_14 = None
        mul_213: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_212, sub_33);  mul_212 = sub_33 = None
        add_95: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_94, mul_213);  add_94 = mul_213 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_49: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_33, [128, 128, 768]);  sum_33 = None
        div_58: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_49, 768);  expand_49 = None
        add_96: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_95, div_58);  add_95 = div_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_5: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_214: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_215: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_96, mul_214);  add_96 = mul_214 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_6: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_216: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_217: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_215, mul_216);  mul_216 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_302: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_217, [16384, 768]);  mul_217 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_16: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_302, permute_175);  permute_175 = None
        permute_176: "f32[768, 16384]" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
        sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_34, [768]);  sum_34 = None
        view_304: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_16, [128, 128, 3072]);  mm_16 = None
        convert_element_type_7: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_218: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_219: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_304, mul_218);  view_304 = mul_218 = None
        view_239: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_64, [128, 128, 3072]);  addmm_64 = None
        mul_159: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_77: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_221: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
        mul_222: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_223: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_222, -0.5);  mul_222 = None
        exp_17: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_223);  mul_223 = None
        mul_224: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_225: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_224);  view_239 = mul_224 = None
        add_98: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_221, mul_225);  mul_221 = mul_225 = None
        mul_226: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_219, add_98);  mul_219 = add_98 = None
        view_305: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_226, [16384, 3072]);  mul_226 = None
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_18: "f32[16384, 768]" = torch.ops.aten.mm.default(view_305, permute_179);  permute_179 = None
        permute_180: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
        sum_35: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        view_306: "f32[3072]" = torch.ops.aten.reshape.default(sum_35, [3072]);  sum_35 = None
        view_307: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_18, [128, 128, 768]);  mm_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_36: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1], True)
        view_308: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        mul_157: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_176, sub_32)
        add_75: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_21, 1e-06)
        div_43: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_157, add_75);  mul_157 = None
        div_60: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_43, add_75);  div_43 = None
        neg_5: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_307)
        mul_227: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_5, div_60);  neg_5 = div_60 = None
        div_61: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_307, add_75);  view_307 = add_75 = None
        sum_37: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
        mul_228: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_61, primals_176);  primals_176 = None
        mul_229: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_32);  div_61 = None
        sum_38: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1], True);  mul_229 = None
        view_309: "f32[768]" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
        neg_6: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_228)
        sum_39: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_6, [2], True);  neg_6 = None
        add_99: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_215, mul_228);  mul_215 = mul_228 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_230: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_21, 2)
        div_62: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_37, mul_230);  sum_37 = mul_230 = None
        eq_14: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_21, 0);  sqrt_21 = None
        where_15: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_14, full_default_13, div_62);  eq_14 = div_62 = None
        mul_231: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_15, 0.002607561929595828);  where_15 = None
        mul_232: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_231, sub_32);  mul_231 = sub_32 = None
        add_100: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_99, mul_232);  add_99 = mul_232 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_50: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_39, [128, 128, 768]);  sum_39 = None
        div_63: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_50, 768);  expand_50 = None
        add_101: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_100, div_63);  add_100 = div_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_8: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_233: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_234: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_101, mul_233);  mul_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_310: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_234, [16384, 768]);  mul_234 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_20: "f32[16384, 768]" = torch.ops.aten.mm.default(view_310, permute_183);  permute_183 = None
        permute_184: "f32[768, 16384]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
        sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        view_311: "f32[768]" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None
        view_312: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_20, [128, 128, 768]);  mm_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_313: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_312, [128, 128, 12, 64]);  view_312 = None
        permute_187: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_61: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_314: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_61, [1536, 128, 64]);  clone_61 = None
        bmm_28: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_188, view_314);  permute_188 = None
        bmm_29: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_314, permute_189);  view_314 = permute_189 = None
        view_315: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_28, [128, 12, 128, 64]);  bmm_28 = None
        view_316: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_29, [128, 12, 128, 128]);  bmm_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_9: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_235: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_236: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_316, mul_235);  view_316 = mul_235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_237: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_236, div_42);  mul_236 = None
        sum_41: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [-1], True)
        neg_7: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_42);  div_42 = None
        fma_1: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_41, mul_237);  neg_7 = sum_41 = mul_237 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_16: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_1);  fma_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_64: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_16, 8.0);  where_16 = None
        view_317: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_64, [1536, 128, 128]);  div_64 = None
        bmm_30: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_190, view_317);  permute_190 = None
        bmm_31: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_317, permute_191);  view_317 = permute_191 = None
        view_318: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_30, [128, 12, 64, 128]);  bmm_30 = None
        view_319: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_31, [128, 12, 128, 64]);  bmm_31 = None
        permute_192: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 2]);  view_318 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_193: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        clone_63: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_320: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_63, [128, 128, 768]);  clone_63 = None
        view_321: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_320, [16384, 768]);  view_320 = None
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_22: "f32[16384, 768]" = torch.ops.aten.mm.default(view_321, permute_194);  permute_194 = None
        permute_195: "f32[768, 16384]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_220);  permute_195 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        view_323: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_22, [128, 128, 768]);  mm_22 = None
        permute_198: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        view_324: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_198, [128, 128, 768]);  permute_198 = None
        clone_64: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_324, memory_format = torch.contiguous_format);  view_324 = None
        view_325: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_64, [16384, 768]);  clone_64 = None
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_24: "f32[16384, 768]" = torch.ops.aten.mm.default(view_325, permute_199);  permute_199 = None
        permute_200: "f32[768, 16384]" = torch.ops.aten.permute.default(view_325, [1, 0])
        mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = None
        sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
        view_326: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
        view_327: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_24, [128, 128, 768]);  mm_24 = None
        add_102: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_323, view_327);  view_323 = view_327 = None
        permute_203: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_65: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_328: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_65, [128, 128, 768]);  clone_65 = None
        view_329: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_328, [16384, 768]);  view_328 = None
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_26: "f32[16384, 768]" = torch.ops.aten.mm.default(view_329, permute_204);  permute_204 = None
        permute_205: "f32[768, 16384]" = torch.ops.aten.permute.default(view_329, [1, 0])
        mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
        sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
        view_330: "f32[768]" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
        view_331: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_26, [128, 128, 768]);  mm_26 = None
        add_103: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_102, view_331);  add_102 = view_331 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_45: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_103, [0, 1], True)
        view_332: "f32[768]" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        mul_152: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_166, sub_30)
        add_72: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_20, 1e-06)
        div_40: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_152, add_72);  mul_152 = None
        div_66: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_40, add_72);  div_40 = None
        neg_8: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_103)
        mul_238: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_8, div_66);  neg_8 = div_66 = None
        div_67: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_103, add_72);  add_103 = add_72 = None
        sum_46: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_67, primals_166);  primals_166 = None
        mul_240: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_30);  div_67 = None
        sum_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 1], True);  mul_240 = None
        view_333: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
        neg_9: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_239)
        sum_48: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True);  neg_9 = None
        add_104: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_101, mul_239);  add_101 = mul_239 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_241: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_20, 2)
        div_68: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_46, mul_241);  sum_46 = mul_241 = None
        eq_15: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_20, 0);  sqrt_20 = None
        where_17: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_15, full_default_13, div_68);  eq_15 = div_68 = None
        mul_242: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_17, 0.002607561929595828);  where_17 = None
        mul_243: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_242, sub_30);  mul_242 = sub_30 = None
        add_105: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_104, mul_243);  add_104 = mul_243 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_51: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_48, [128, 128, 768]);  sum_48 = None
        div_69: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_51, 768);  expand_51 = None
        add_106: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_105, div_69);  add_105 = div_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_10: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_244: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_245: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_106, mul_244);  add_106 = mul_244 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_11: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_246: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_247: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_245, mul_246);  mul_246 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_334: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_247, [16384, 768]);  mul_247 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_28: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_334, permute_208);  permute_208 = None
        permute_209: "f32[768, 16384]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
        sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        view_335: "f32[768]" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
        view_336: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_28, [128, 128, 3072]);  mm_28 = None
        convert_element_type_12: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_248: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_249: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_336, mul_248);  view_336 = mul_248 = None
        view_217: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_58, [128, 128, 3072]);  addmm_58 = None
        mul_144: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_70: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_251: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
        mul_252: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_253: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_252, -0.5);  mul_252 = None
        exp_18: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_253);  mul_253 = None
        mul_254: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_255: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_254);  view_217 = mul_254 = None
        add_108: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_251, mul_255);  mul_251 = mul_255 = None
        mul_256: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_249, add_108);  mul_249 = add_108 = None
        view_337: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_256, [16384, 3072]);  mul_256 = None
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_30: "f32[16384, 768]" = torch.ops.aten.mm.default(view_337, permute_212);  permute_212 = None
        permute_213: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
        sum_50: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        view_338: "f32[3072]" = torch.ops.aten.reshape.default(sum_50, [3072]);  sum_50 = None
        view_339: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_30, [128, 128, 768]);  mm_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_51: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1], True)
        view_340: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        mul_142: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_160, sub_29)
        add_68: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_19, 1e-06)
        div_39: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_142, add_68);  mul_142 = None
        div_71: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_39, add_68);  div_39 = None
        neg_10: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_339)
        mul_257: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_10, div_71);  neg_10 = div_71 = None
        div_72: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_339, add_68);  view_339 = add_68 = None
        sum_52: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
        mul_258: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_72, primals_160);  primals_160 = None
        mul_259: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_29);  div_72 = None
        sum_53: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1], True);  mul_259 = None
        view_341: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
        neg_11: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_258)
        sum_54: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_11, [2], True);  neg_11 = None
        add_109: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_245, mul_258);  mul_245 = mul_258 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_260: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_19, 2)
        div_73: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_52, mul_260);  sum_52 = mul_260 = None
        eq_16: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_19, 0);  sqrt_19 = None
        where_18: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_16, full_default_13, div_73);  eq_16 = div_73 = None
        mul_261: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_18, 0.002607561929595828);  where_18 = None
        mul_262: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_261, sub_29);  mul_261 = sub_29 = None
        add_110: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_109, mul_262);  add_109 = mul_262 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_52: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_54, [128, 128, 768]);  sum_54 = None
        div_74: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_52, 768);  expand_52 = None
        add_111: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_110, div_74);  add_110 = div_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_13: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_263: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_264: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_111, mul_263);  mul_263 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_342: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_264, [16384, 768]);  mul_264 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_32: "f32[16384, 768]" = torch.ops.aten.mm.default(view_342, permute_216);  permute_216 = None
        permute_217: "f32[768, 16384]" = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
        sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_342, [0], True);  view_342 = None
        view_343: "f32[768]" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        view_344: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_32, [128, 128, 768]);  mm_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_345: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_344, [128, 128, 12, 64]);  view_344 = None
        permute_220: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_70: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_346: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_70, [1536, 128, 64]);  clone_70 = None
        bmm_32: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_221, view_346);  permute_221 = None
        bmm_33: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_346, permute_222);  view_346 = permute_222 = None
        view_347: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_32, [128, 12, 128, 64]);  bmm_32 = None
        view_348: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, [128, 12, 128, 128]);  bmm_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_14: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_265: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_266: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_348, mul_265);  view_348 = mul_265 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_267: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_266, div_38);  mul_266 = None
        sum_56: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [-1], True)
        neg_12: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_38);  div_38 = None
        fma_2: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_56, mul_267);  neg_12 = sum_56 = mul_267 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_19: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_2);  fma_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_75: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_19, 8.0);  where_19 = None
        view_349: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_75, [1536, 128, 128]);  div_75 = None
        bmm_34: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_223, view_349);  permute_223 = None
        bmm_35: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_349, permute_224);  view_349 = permute_224 = None
        view_350: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_34, [128, 12, 64, 128]);  bmm_34 = None
        view_351: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, [128, 12, 128, 64]);  bmm_35 = None
        permute_225: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2]);  view_350 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_226: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        clone_72: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_352: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_72, [128, 128, 768]);  clone_72 = None
        view_353: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_352, [16384, 768]);  view_352 = None
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_34: "f32[16384, 768]" = torch.ops.aten.mm.default(view_353, permute_227);  permute_227 = None
        permute_228: "f32[768, 16384]" = torch.ops.aten.permute.default(view_353, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_198);  permute_228 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
        view_354: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_355: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_34, [128, 128, 768]);  mm_34 = None
        permute_231: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_356: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_231, [128, 128, 768]);  permute_231 = None
        clone_73: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_356, memory_format = torch.contiguous_format);  view_356 = None
        view_357: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_73, [16384, 768]);  clone_73 = None
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_36: "f32[16384, 768]" = torch.ops.aten.mm.default(view_357, permute_232);  permute_232 = None
        permute_233: "f32[768, 16384]" = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = None
        sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
        view_358: "f32[768]" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
        view_359: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_36, [128, 128, 768]);  mm_36 = None
        add_112: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_355, view_359);  view_355 = view_359 = None
        permute_236: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        clone_74: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_360: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_74, [128, 128, 768]);  clone_74 = None
        view_361: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_360, [16384, 768]);  view_360 = None
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_38: "f32[16384, 768]" = torch.ops.aten.mm.default(view_361, permute_237);  permute_237 = None
        permute_238: "f32[768, 16384]" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
        sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        view_362: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
        view_363: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_38, [128, 128, 768]);  mm_38 = None
        add_113: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_112, view_363);  add_112 = view_363 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_60: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1], True)
        view_364: "f32[768]" = torch.ops.aten.reshape.default(sum_60, [768]);  sum_60 = None
        mul_137: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_150, sub_27)
        add_65: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_18, 1e-06)
        div_36: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_137, add_65);  mul_137 = None
        div_77: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_36, add_65);  div_36 = None
        neg_13: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_113)
        mul_268: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_13, div_77);  neg_13 = div_77 = None
        div_78: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_113, add_65);  add_113 = add_65 = None
        sum_61: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
        mul_269: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_78, primals_150);  primals_150 = None
        mul_270: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_78, sub_27);  div_78 = None
        sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1], True);  mul_270 = None
        view_365: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
        neg_14: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_269)
        sum_63: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_14, [2], True);  neg_14 = None
        add_114: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_111, mul_269);  add_111 = mul_269 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_271: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_18, 2)
        div_79: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_61, mul_271);  sum_61 = mul_271 = None
        eq_17: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_18, 0);  sqrt_18 = None
        where_20: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_17, full_default_13, div_79);  eq_17 = div_79 = None
        mul_272: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_20, 0.002607561929595828);  where_20 = None
        mul_273: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_272, sub_27);  mul_272 = sub_27 = None
        add_115: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_114, mul_273);  add_114 = mul_273 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_53: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_63, [128, 128, 768]);  sum_63 = None
        div_80: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_53, 768);  expand_53 = None
        add_116: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_115, div_80);  add_115 = div_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_15: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_274: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_275: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_274);  add_116 = mul_274 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_16: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_276: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_277: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_275, mul_276);  mul_276 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_366: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_277, [16384, 768]);  mul_277 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_40: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_366, permute_241);  permute_241 = None
        permute_242: "f32[768, 16384]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
        sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        view_367: "f32[768]" = torch.ops.aten.reshape.default(sum_64, [768]);  sum_64 = None
        view_368: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_40, [128, 128, 3072]);  mm_40 = None
        convert_element_type_17: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_278: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_279: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_368, mul_278);  view_368 = mul_278 = None
        view_195: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_52, [128, 128, 3072]);  addmm_52 = None
        mul_129: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_63: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_281: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
        mul_282: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_283: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_282, -0.5);  mul_282 = None
        exp_19: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_283);  mul_283 = None
        mul_284: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_285: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_284);  view_195 = mul_284 = None
        add_118: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_281, mul_285);  mul_281 = mul_285 = None
        mul_286: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_279, add_118);  mul_279 = add_118 = None
        view_369: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_286, [16384, 3072]);  mul_286 = None
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_42: "f32[16384, 768]" = torch.ops.aten.mm.default(view_369, permute_245);  permute_245 = None
        permute_246: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
        sum_65: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        view_370: "f32[3072]" = torch.ops.aten.reshape.default(sum_65, [3072]);  sum_65 = None
        view_371: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_42, [128, 128, 768]);  mm_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_66: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0, 1], True)
        view_372: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        mul_127: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_144, sub_26)
        add_61: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_17, 1e-06)
        div_35: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_127, add_61);  mul_127 = None
        div_82: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_35, add_61);  div_35 = None
        neg_15: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_371)
        mul_287: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_15, div_82);  neg_15 = div_82 = None
        div_83: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_371, add_61);  view_371 = add_61 = None
        sum_67: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
        mul_288: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_83, primals_144);  primals_144 = None
        mul_289: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_83, sub_26);  div_83 = None
        sum_68: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1], True);  mul_289 = None
        view_373: "f32[768]" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
        neg_16: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_288)
        sum_69: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_16, [2], True);  neg_16 = None
        add_119: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_275, mul_288);  mul_275 = mul_288 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_290: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_17, 2)
        div_84: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_67, mul_290);  sum_67 = mul_290 = None
        eq_18: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_17, 0);  sqrt_17 = None
        where_21: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_18, full_default_13, div_84);  eq_18 = div_84 = None
        mul_291: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_21, 0.002607561929595828);  where_21 = None
        mul_292: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_291, sub_26);  mul_291 = sub_26 = None
        add_120: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_119, mul_292);  add_119 = mul_292 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_54: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_69, [128, 128, 768]);  sum_69 = None
        div_85: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_54, 768);  expand_54 = None
        add_121: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_120, div_85);  add_120 = div_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_18: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_293: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_294: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_293);  mul_293 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_374: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_294, [16384, 768]);  mul_294 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_44: "f32[16384, 768]" = torch.ops.aten.mm.default(view_374, permute_249);  permute_249 = None
        permute_250: "f32[768, 16384]" = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
        sum_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
        view_375: "f32[768]" = torch.ops.aten.reshape.default(sum_70, [768]);  sum_70 = None
        view_376: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_44, [128, 128, 768]);  mm_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_377: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_376, [128, 128, 12, 64]);  view_376 = None
        permute_253: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_79: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_378: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_79, [1536, 128, 64]);  clone_79 = None
        bmm_36: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_254, view_378);  permute_254 = None
        bmm_37: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_378, permute_255);  view_378 = permute_255 = None
        view_379: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_36, [128, 12, 128, 64]);  bmm_36 = None
        view_380: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_37, [128, 12, 128, 128]);  bmm_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_19: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_295: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_296: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_380, mul_295);  view_380 = mul_295 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_297: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_296, div_34);  mul_296 = None
        sum_71: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [-1], True)
        neg_17: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_3: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_71, mul_297);  neg_17 = sum_71 = mul_297 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_22: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_3);  fma_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_86: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_22, 8.0);  where_22 = None
        view_381: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_86, [1536, 128, 128]);  div_86 = None
        bmm_38: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_256, view_381);  permute_256 = None
        bmm_39: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_381, permute_257);  view_381 = permute_257 = None
        view_382: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_38, [128, 12, 64, 128]);  bmm_38 = None
        view_383: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_39, [128, 12, 128, 64]);  bmm_39 = None
        permute_258: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_382, [0, 1, 3, 2]);  view_382 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_259: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        clone_81: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_384: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_81, [128, 128, 768]);  clone_81 = None
        view_385: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_384, [16384, 768]);  view_384 = None
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_46: "f32[16384, 768]" = torch.ops.aten.mm.default(view_385, permute_260);  permute_260 = None
        permute_261: "f32[768, 16384]" = torch.ops.aten.permute.default(view_385, [1, 0])
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view_176);  permute_261 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
        view_386: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        view_387: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_46, [128, 128, 768]);  mm_46 = None
        permute_264: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        view_388: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_264, [128, 128, 768]);  permute_264 = None
        clone_82: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_388, memory_format = torch.contiguous_format);  view_388 = None
        view_389: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_82, [16384, 768]);  clone_82 = None
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_48: "f32[16384, 768]" = torch.ops.aten.mm.default(view_389, permute_265);  permute_265 = None
        permute_266: "f32[768, 16384]" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = None
        sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        view_390: "f32[768]" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
        view_391: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_48, [128, 128, 768]);  mm_48 = None
        add_122: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_387, view_391);  view_387 = view_391 = None
        permute_269: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_83: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_392: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_83, [128, 128, 768]);  clone_83 = None
        view_393: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_392, [16384, 768]);  view_392 = None
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_50: "f32[16384, 768]" = torch.ops.aten.mm.default(view_393, permute_270);  permute_270 = None
        permute_271: "f32[768, 16384]" = torch.ops.aten.permute.default(view_393, [1, 0])
        mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
        sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_393, [0], True);  view_393 = None
        view_394: "f32[768]" = torch.ops.aten.reshape.default(sum_74, [768]);  sum_74 = None
        view_395: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_50, [128, 128, 768]);  mm_50 = None
        add_123: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_122, view_395);  add_122 = view_395 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_75: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1], True)
        view_396: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        mul_122: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_134, sub_24)
        add_58: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_16, 1e-06)
        div_32: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_122, add_58);  mul_122 = None
        div_88: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_32, add_58);  div_32 = None
        neg_18: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_123)
        mul_298: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_18, div_88);  neg_18 = div_88 = None
        div_89: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_123, add_58);  add_123 = add_58 = None
        sum_76: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_89, primals_134);  primals_134 = None
        mul_300: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_89, sub_24);  div_89 = None
        sum_77: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1], True);  mul_300 = None
        view_397: "f32[768]" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
        neg_19: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_299)
        sum_78: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_19, [2], True);  neg_19 = None
        add_124: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_121, mul_299);  add_121 = mul_299 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_301: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_16, 2)
        div_90: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_76, mul_301);  sum_76 = mul_301 = None
        eq_19: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_16, 0);  sqrt_16 = None
        where_23: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_19, full_default_13, div_90);  eq_19 = div_90 = None
        mul_302: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_23, 0.002607561929595828);  where_23 = None
        mul_303: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_302, sub_24);  mul_302 = sub_24 = None
        add_125: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_124, mul_303);  add_124 = mul_303 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_55: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_78, [128, 128, 768]);  sum_78 = None
        div_91: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_55, 768);  expand_55 = None
        add_126: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_125, div_91);  add_125 = div_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_20: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_304: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_305: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_304);  add_126 = mul_304 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_21: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_306: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_307: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_306 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_398: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_307, [16384, 768]);  mul_307 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_274: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_52: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_398, permute_274);  permute_274 = None
        permute_275: "f32[768, 16384]" = torch.ops.aten.permute.default(view_398, [1, 0])
        mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
        sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
        view_399: "f32[768]" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
        view_400: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_52, [128, 128, 3072]);  mm_52 = None
        convert_element_type_22: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_308: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_309: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_400, mul_308);  view_400 = mul_308 = None
        view_173: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_46, [128, 128, 3072]);  addmm_46 = None
        mul_114: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
        add_56: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_311: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_312: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_313: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_312, -0.5);  mul_312 = None
        exp_20: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_313);  mul_313 = None
        mul_314: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_315: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_314);  view_173 = mul_314 = None
        add_128: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_311, mul_315);  mul_311 = mul_315 = None
        mul_316: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_309, add_128);  mul_309 = add_128 = None
        view_401: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_316, [16384, 3072]);  mul_316 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_54: "f32[16384, 768]" = torch.ops.aten.mm.default(view_401, permute_278);  permute_278 = None
        permute_279: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_401, [1, 0])
        mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
        sum_80: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
        view_402: "f32[3072]" = torch.ops.aten.reshape.default(sum_80, [3072]);  sum_80 = None
        view_403: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_54, [128, 128, 768]);  mm_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_81: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0, 1], True)
        view_404: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        mul_112: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_128, sub_23)
        add_54: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_15, 1e-06)
        div_31: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_112, add_54);  mul_112 = None
        div_93: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_31, add_54);  div_31 = None
        neg_20: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_403)
        mul_317: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_20, div_93);  neg_20 = div_93 = None
        div_94: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_403, add_54);  view_403 = add_54 = None
        sum_82: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
        mul_318: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_94, primals_128);  primals_128 = None
        mul_319: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_94, sub_23);  div_94 = None
        sum_83: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1], True);  mul_319 = None
        view_405: "f32[768]" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        neg_21: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_318)
        sum_84: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True);  neg_21 = None
        add_129: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_305, mul_318);  mul_305 = mul_318 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_320: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_15, 2)
        div_95: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_82, mul_320);  sum_82 = mul_320 = None
        eq_20: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_15, 0);  sqrt_15 = None
        where_24: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_20, full_default_13, div_95);  eq_20 = div_95 = None
        mul_321: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_24, 0.002607561929595828);  where_24 = None
        mul_322: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_321, sub_23);  mul_321 = sub_23 = None
        add_130: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_129, mul_322);  add_129 = mul_322 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_56: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_84, [128, 128, 768]);  sum_84 = None
        div_96: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_56, 768);  expand_56 = None
        add_131: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_130, div_96);  add_130 = div_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_23: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_323: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_324: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_323);  mul_323 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_406: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_324, [16384, 768]);  mul_324 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_56: "f32[16384, 768]" = torch.ops.aten.mm.default(view_406, permute_282);  permute_282 = None
        permute_283: "f32[768, 16384]" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
        sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        view_407: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
        view_408: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_56, [128, 128, 768]);  mm_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_409: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_408, [128, 128, 12, 64]);  view_408 = None
        permute_286: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_88: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_410: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_88, [1536, 128, 64]);  clone_88 = None
        bmm_40: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_287, view_410);  permute_287 = None
        bmm_41: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_410, permute_288);  view_410 = permute_288 = None
        view_411: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_40, [128, 12, 128, 64]);  bmm_40 = None
        view_412: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_41, [128, 12, 128, 128]);  bmm_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_24: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_325: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_326: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_412, mul_325);  view_412 = mul_325 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_327: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_326, div_30);  mul_326 = None
        sum_86: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [-1], True)
        neg_22: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_4: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_86, mul_327);  neg_22 = sum_86 = mul_327 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_25: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_4);  fma_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_97: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_25, 8.0);  where_25 = None
        view_413: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_97, [1536, 128, 128]);  div_97 = None
        bmm_42: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_289, view_413);  permute_289 = None
        bmm_43: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_413, permute_290);  view_413 = permute_290 = None
        view_414: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_42, [128, 12, 64, 128]);  bmm_42 = None
        view_415: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_43, [128, 12, 128, 64]);  bmm_43 = None
        permute_291: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_414, [0, 1, 3, 2]);  view_414 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_292: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        clone_90: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_416: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_90, [128, 128, 768]);  clone_90 = None
        view_417: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_416, [16384, 768]);  view_416 = None
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_293: "f32[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_58: "f32[16384, 768]" = torch.ops.aten.mm.default(view_417, permute_293);  permute_293 = None
        permute_294: "f32[768, 16384]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_294, view_154);  permute_294 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
        view_418: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_419: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_58, [128, 128, 768]);  mm_58 = None
        permute_297: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
        view_420: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_297, [128, 128, 768]);  permute_297 = None
        clone_91: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_420, memory_format = torch.contiguous_format);  view_420 = None
        view_421: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_91, [16384, 768]);  clone_91 = None
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_60: "f32[16384, 768]" = torch.ops.aten.mm.default(view_421, permute_298);  permute_298 = None
        permute_299: "f32[768, 16384]" = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = None
        sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        view_422: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
        view_423: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_60, [128, 128, 768]);  mm_60 = None
        add_132: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_419, view_423);  view_419 = view_423 = None
        permute_302: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
        clone_92: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_424: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_92, [128, 128, 768]);  clone_92 = None
        view_425: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_424, [16384, 768]);  view_424 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_62: "f32[16384, 768]" = torch.ops.aten.mm.default(view_425, permute_303);  permute_303 = None
        permute_304: "f32[768, 16384]" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
        sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        view_426: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
        view_427: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_62, [128, 128, 768]);  mm_62 = None
        add_133: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_132, view_427);  add_132 = view_427 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_90: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1], True)
        view_428: "f32[768]" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
        mul_107: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_118, sub_21)
        add_51: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_14, 1e-06)
        div_28: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_107, add_51);  mul_107 = None
        div_99: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_28, add_51);  div_28 = None
        neg_23: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_133)
        mul_328: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_23, div_99);  neg_23 = div_99 = None
        div_100: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_133, add_51);  add_133 = add_51 = None
        sum_91: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_100, primals_118);  primals_118 = None
        mul_330: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_100, sub_21);  div_100 = None
        sum_92: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1], True);  mul_330 = None
        view_429: "f32[768]" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
        neg_24: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_329)
        sum_93: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_24, [2], True);  neg_24 = None
        add_134: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_131, mul_329);  add_131 = mul_329 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_331: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_14, 2)
        div_101: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_91, mul_331);  sum_91 = mul_331 = None
        eq_21: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_14, 0);  sqrt_14 = None
        where_26: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_21, full_default_13, div_101);  eq_21 = div_101 = None
        mul_332: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_26, 0.002607561929595828);  where_26 = None
        mul_333: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_332, sub_21);  mul_332 = sub_21 = None
        add_135: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_134, mul_333);  add_134 = mul_333 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_57: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_93, [128, 128, 768]);  sum_93 = None
        div_102: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_57, 768);  expand_57 = None
        add_136: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_135, div_102);  add_135 = div_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_25: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_334: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_335: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_136, mul_334);  add_136 = mul_334 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_26: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_336: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_337: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_335, mul_336);  mul_336 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_430: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_337, [16384, 768]);  mul_337 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_64: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_430, permute_307);  permute_307 = None
        permute_308: "f32[768, 16384]" = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
        sum_94: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        view_431: "f32[768]" = torch.ops.aten.reshape.default(sum_94, [768]);  sum_94 = None
        view_432: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_64, [128, 128, 3072]);  mm_64 = None
        convert_element_type_27: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_338: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_339: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_432, mul_338);  view_432 = mul_338 = None
        view_151: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_40, [128, 128, 3072]);  addmm_40 = None
        mul_99: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_49: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_341: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_342: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_343: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_342, -0.5);  mul_342 = None
        exp_21: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_343);  mul_343 = None
        mul_344: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_345: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_344);  view_151 = mul_344 = None
        add_138: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_341, mul_345);  mul_341 = mul_345 = None
        mul_346: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_339, add_138);  mul_339 = add_138 = None
        view_433: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_346, [16384, 3072]);  mul_346 = None
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_66: "f32[16384, 768]" = torch.ops.aten.mm.default(view_433, permute_311);  permute_311 = None
        permute_312: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
        sum_95: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
        view_434: "f32[3072]" = torch.ops.aten.reshape.default(sum_95, [3072]);  sum_95 = None
        view_435: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_66, [128, 128, 768]);  mm_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_96: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0, 1], True)
        view_436: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        mul_97: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_112, sub_20)
        add_47: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_13, 1e-06)
        div_27: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_97, add_47);  mul_97 = None
        div_104: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_27, add_47);  div_27 = None
        neg_25: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_435)
        mul_347: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_25, div_104);  neg_25 = div_104 = None
        div_105: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_435, add_47);  view_435 = add_47 = None
        sum_97: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, primals_112);  primals_112 = None
        mul_349: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, sub_20);  div_105 = None
        sum_98: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1], True);  mul_349 = None
        view_437: "f32[768]" = torch.ops.aten.reshape.default(sum_98, [768]);  sum_98 = None
        neg_26: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_348)
        sum_99: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_26, [2], True);  neg_26 = None
        add_139: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_335, mul_348);  mul_335 = mul_348 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_350: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_13, 2)
        div_106: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_97, mul_350);  sum_97 = mul_350 = None
        eq_22: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_13, 0);  sqrt_13 = None
        where_27: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_22, full_default_13, div_106);  eq_22 = div_106 = None
        mul_351: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_27, 0.002607561929595828);  where_27 = None
        mul_352: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_351, sub_20);  mul_351 = sub_20 = None
        add_140: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_139, mul_352);  add_139 = mul_352 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_58: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_99, [128, 128, 768]);  sum_99 = None
        div_107: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_58, 768);  expand_58 = None
        add_141: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_140, div_107);  add_140 = div_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_28: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_353: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_354: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_353);  mul_353 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_438: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_354, [16384, 768]);  mul_354 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_68: "f32[16384, 768]" = torch.ops.aten.mm.default(view_438, permute_315);  permute_315 = None
        permute_316: "f32[768, 16384]" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
        sum_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
        view_439: "f32[768]" = torch.ops.aten.reshape.default(sum_100, [768]);  sum_100 = None
        view_440: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_68, [128, 128, 768]);  mm_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_441: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_440, [128, 128, 12, 64]);  view_440 = None
        permute_319: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_97: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_442: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_97, [1536, 128, 64]);  clone_97 = None
        bmm_44: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_320, view_442);  permute_320 = None
        bmm_45: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_442, permute_321);  view_442 = permute_321 = None
        view_443: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_44, [128, 12, 128, 64]);  bmm_44 = None
        view_444: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_45, [128, 12, 128, 128]);  bmm_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_29: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_355: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_356: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_444, mul_355);  view_444 = mul_355 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_357: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_356, div_26);  mul_356 = None
        sum_101: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [-1], True)
        neg_27: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_26);  div_26 = None
        fma_5: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_27, sum_101, mul_357);  neg_27 = sum_101 = mul_357 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_28: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_5);  fma_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_108: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_28, 8.0);  where_28 = None
        view_445: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_108, [1536, 128, 128]);  div_108 = None
        bmm_46: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_322, view_445);  permute_322 = None
        bmm_47: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_445, permute_323);  view_445 = permute_323 = None
        view_446: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_46, [128, 12, 64, 128]);  bmm_46 = None
        view_447: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_47, [128, 12, 128, 64]);  bmm_47 = None
        permute_324: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_446, [0, 1, 3, 2]);  view_446 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_325: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        clone_99: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_448: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_99, [128, 128, 768]);  clone_99 = None
        view_449: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_448, [16384, 768]);  view_448 = None
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_326: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_70: "f32[16384, 768]" = torch.ops.aten.mm.default(view_449, permute_326);  permute_326 = None
        permute_327: "f32[768, 16384]" = torch.ops.aten.permute.default(view_449, [1, 0])
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_327, view_132);  permute_327 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
        view_450: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        view_451: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_70, [128, 128, 768]);  mm_70 = None
        permute_330: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
        view_452: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_330, [128, 128, 768]);  permute_330 = None
        clone_100: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_452, memory_format = torch.contiguous_format);  view_452 = None
        view_453: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_100, [16384, 768]);  clone_100 = None
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_72: "f32[16384, 768]" = torch.ops.aten.mm.default(view_453, permute_331);  permute_331 = None
        permute_332: "f32[768, 16384]" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = None
        sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        view_454: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
        view_455: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_72, [128, 128, 768]);  mm_72 = None
        add_142: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_451, view_455);  view_451 = view_455 = None
        permute_335: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        clone_101: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_456: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_101, [128, 128, 768]);  clone_101 = None
        view_457: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_456, [16384, 768]);  view_456 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_74: "f32[16384, 768]" = torch.ops.aten.mm.default(view_457, permute_336);  permute_336 = None
        permute_337: "f32[768, 16384]" = torch.ops.aten.permute.default(view_457, [1, 0])
        mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
        sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
        view_458: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
        view_459: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_74, [128, 128, 768]);  mm_74 = None
        add_143: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_142, view_459);  add_142 = view_459 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_105: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1], True)
        view_460: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        mul_92: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_102, sub_18)
        add_44: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_12, 1e-06)
        div_24: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_92, add_44);  mul_92 = None
        div_110: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_24, add_44);  div_24 = None
        neg_28: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_143)
        mul_358: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_28, div_110);  neg_28 = div_110 = None
        div_111: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_143, add_44);  add_143 = add_44 = None
        sum_106: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_111, primals_102);  primals_102 = None
        mul_360: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_111, sub_18);  div_111 = None
        sum_107: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1], True);  mul_360 = None
        view_461: "f32[768]" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
        neg_29: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_359)
        sum_108: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_29, [2], True);  neg_29 = None
        add_144: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_141, mul_359);  add_141 = mul_359 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_361: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_12, 2)
        div_112: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_106, mul_361);  sum_106 = mul_361 = None
        eq_23: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_12, 0);  sqrt_12 = None
        where_29: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_23, full_default_13, div_112);  eq_23 = div_112 = None
        mul_362: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_29, 0.002607561929595828);  where_29 = None
        mul_363: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_362, sub_18);  mul_362 = sub_18 = None
        add_145: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_144, mul_363);  add_144 = mul_363 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_59: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_108, [128, 128, 768]);  sum_108 = None
        div_113: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_59, 768);  expand_59 = None
        add_146: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_145, div_113);  add_145 = div_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_30: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_364: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_365: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_364);  add_146 = mul_364 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_31: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_366: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_367: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_365, mul_366);  mul_366 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_462: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_367, [16384, 768]);  mul_367 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_340: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_76: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_462, permute_340);  permute_340 = None
        permute_341: "f32[768, 16384]" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
        sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
        view_463: "f32[768]" = torch.ops.aten.reshape.default(sum_109, [768]);  sum_109 = None
        view_464: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_76, [128, 128, 3072]);  mm_76 = None
        convert_element_type_32: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_368: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_369: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_464, mul_368);  view_464 = mul_368 = None
        view_129: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, [128, 128, 3072]);  addmm_34 = None
        mul_84: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_42: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_371: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_372: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_373: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_372, -0.5);  mul_372 = None
        exp_22: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_373);  mul_373 = None
        mul_374: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_375: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_374);  view_129 = mul_374 = None
        add_148: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_371, mul_375);  mul_371 = mul_375 = None
        mul_376: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_369, add_148);  mul_369 = add_148 = None
        view_465: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_376, [16384, 3072]);  mul_376 = None
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_78: "f32[16384, 768]" = torch.ops.aten.mm.default(view_465, permute_344);  permute_344 = None
        permute_345: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_465, [1, 0])
        mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
        sum_110: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
        view_466: "f32[3072]" = torch.ops.aten.reshape.default(sum_110, [3072]);  sum_110 = None
        view_467: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_78, [128, 128, 768]);  mm_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_111: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0, 1], True)
        view_468: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        mul_82: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_96, sub_17)
        add_40: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_11, 1e-06)
        div_23: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_82, add_40);  mul_82 = None
        div_115: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_23, add_40);  div_23 = None
        neg_30: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_467)
        mul_377: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_30, div_115);  neg_30 = div_115 = None
        div_116: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_467, add_40);  view_467 = add_40 = None
        sum_112: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
        mul_378: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_116, primals_96);  primals_96 = None
        mul_379: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_116, sub_17);  div_116 = None
        sum_113: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1], True);  mul_379 = None
        view_469: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
        neg_31: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_378)
        sum_114: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_31, [2], True);  neg_31 = None
        add_149: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_365, mul_378);  mul_365 = mul_378 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_380: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_11, 2)
        div_117: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_112, mul_380);  sum_112 = mul_380 = None
        eq_24: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_11, 0);  sqrt_11 = None
        where_30: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_24, full_default_13, div_117);  eq_24 = div_117 = None
        mul_381: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_30, 0.002607561929595828);  where_30 = None
        mul_382: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_381, sub_17);  mul_381 = sub_17 = None
        add_150: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_149, mul_382);  add_149 = mul_382 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_60: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_114, [128, 128, 768]);  sum_114 = None
        div_118: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_60, 768);  expand_60 = None
        add_151: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_150, div_118);  add_150 = div_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_33: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_383: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_384: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_151, mul_383);  mul_383 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_470: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_384, [16384, 768]);  mul_384 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_348: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_80: "f32[16384, 768]" = torch.ops.aten.mm.default(view_470, permute_348);  permute_348 = None
        permute_349: "f32[768, 16384]" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
        sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        view_471: "f32[768]" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        view_472: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_80, [128, 128, 768]);  mm_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_473: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_472, [128, 128, 12, 64]);  view_472 = None
        permute_352: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_106: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_474: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_106, [1536, 128, 64]);  clone_106 = None
        bmm_48: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_353, view_474);  permute_353 = None
        bmm_49: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_474, permute_354);  view_474 = permute_354 = None
        view_475: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_48, [128, 12, 128, 64]);  bmm_48 = None
        view_476: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_49, [128, 12, 128, 128]);  bmm_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_34: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_385: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_386: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_476, mul_385);  view_476 = mul_385 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_387: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_386, div_22);  mul_386 = None
        sum_116: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [-1], True)
        neg_32: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_6: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_32, sum_116, mul_387);  neg_32 = sum_116 = mul_387 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_31: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_6);  fma_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_119: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_31, 8.0);  where_31 = None
        view_477: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_119, [1536, 128, 128]);  div_119 = None
        bmm_50: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_355, view_477);  permute_355 = None
        bmm_51: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_477, permute_356);  view_477 = permute_356 = None
        view_478: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_50, [128, 12, 64, 128]);  bmm_50 = None
        view_479: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_51, [128, 12, 128, 64]);  bmm_51 = None
        permute_357: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_478, [0, 1, 3, 2]);  view_478 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_358: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
        clone_108: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_480: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_108, [128, 128, 768]);  clone_108 = None
        view_481: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_480, [16384, 768]);  view_480 = None
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_359: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_82: "f32[16384, 768]" = torch.ops.aten.mm.default(view_481, permute_359);  permute_359 = None
        permute_360: "f32[768, 16384]" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_360, view_110);  permute_360 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        view_482: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_483: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_82, [128, 128, 768]);  mm_82 = None
        permute_363: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
        view_484: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_363, [128, 128, 768]);  permute_363 = None
        clone_109: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_484, memory_format = torch.contiguous_format);  view_484 = None
        view_485: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_109, [16384, 768]);  clone_109 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_84: "f32[16384, 768]" = torch.ops.aten.mm.default(view_485, permute_364);  permute_364 = None
        permute_365: "f32[768, 16384]" = torch.ops.aten.permute.default(view_485, [1, 0])
        mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = None
        sum_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
        view_486: "f32[768]" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
        view_487: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_84, [128, 128, 768]);  mm_84 = None
        add_152: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_483, view_487);  view_483 = view_487 = None
        permute_368: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        clone_110: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_488: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_110, [128, 128, 768]);  clone_110 = None
        view_489: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_488, [16384, 768]);  view_488 = None
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_86: "f32[16384, 768]" = torch.ops.aten.mm.default(view_489, permute_369);  permute_369 = None
        permute_370: "f32[768, 16384]" = torch.ops.aten.permute.default(view_489, [1, 0])
        mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
        sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
        view_490: "f32[768]" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
        view_491: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_86, [128, 128, 768]);  mm_86 = None
        add_153: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_152, view_491);  add_152 = view_491 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_120: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1], True)
        view_492: "f32[768]" = torch.ops.aten.reshape.default(sum_120, [768]);  sum_120 = None
        mul_77: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_86, sub_15)
        add_37: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_10, 1e-06)
        div_20: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_77, add_37);  mul_77 = None
        div_121: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_20, add_37);  div_20 = None
        neg_33: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_153)
        mul_388: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_33, div_121);  neg_33 = div_121 = None
        div_122: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_153, add_37);  add_153 = add_37 = None
        sum_121: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, primals_86);  primals_86 = None
        mul_390: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, sub_15);  div_122 = None
        sum_122: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1], True);  mul_390 = None
        view_493: "f32[768]" = torch.ops.aten.reshape.default(sum_122, [768]);  sum_122 = None
        neg_34: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_389)
        sum_123: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_34, [2], True);  neg_34 = None
        add_154: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_151, mul_389);  add_151 = mul_389 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_391: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_10, 2)
        div_123: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_121, mul_391);  sum_121 = mul_391 = None
        eq_25: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_10, 0);  sqrt_10 = None
        where_32: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_25, full_default_13, div_123);  eq_25 = div_123 = None
        mul_392: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_32, 0.002607561929595828);  where_32 = None
        mul_393: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_392, sub_15);  mul_392 = sub_15 = None
        add_155: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_154, mul_393);  add_154 = mul_393 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_61: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_123, [128, 128, 768]);  sum_123 = None
        div_124: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_61, 768);  expand_61 = None
        add_156: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_155, div_124);  add_155 = div_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_35: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_394: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_395: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_394);  add_156 = mul_394 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_36: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_396: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_397: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_395, mul_396);  mul_396 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_494: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_397, [16384, 768]);  mul_397 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_373: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_88: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_494, permute_373);  permute_373 = None
        permute_374: "f32[768, 16384]" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
        sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
        view_495: "f32[768]" = torch.ops.aten.reshape.default(sum_124, [768]);  sum_124 = None
        view_496: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_88, [128, 128, 3072]);  mm_88 = None
        convert_element_type_37: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_398: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_399: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_496, mul_398);  view_496 = mul_398 = None
        view_107: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_28, [128, 128, 3072]);  addmm_28 = None
        mul_69: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_35: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_401: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
        mul_402: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_403: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_402, -0.5);  mul_402 = None
        exp_23: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_403);  mul_403 = None
        mul_404: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_405: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_404);  view_107 = mul_404 = None
        add_158: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
        mul_406: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_399, add_158);  mul_399 = add_158 = None
        view_497: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_406, [16384, 3072]);  mul_406 = None
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_90: "f32[16384, 768]" = torch.ops.aten.mm.default(view_497, permute_377);  permute_377 = None
        permute_378: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_497, [1, 0])
        mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
        sum_125: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
        view_498: "f32[3072]" = torch.ops.aten.reshape.default(sum_125, [3072]);  sum_125 = None
        view_499: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_90, [128, 128, 768]);  mm_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_499, [0, 1], True)
        view_500: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        mul_67: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_80, sub_14)
        add_33: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_9, 1e-06)
        div_19: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_67, add_33);  mul_67 = None
        div_126: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_19, add_33);  div_19 = None
        neg_35: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_499)
        mul_407: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_35, div_126);  neg_35 = div_126 = None
        div_127: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_499, add_33);  view_499 = add_33 = None
        sum_127: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True);  mul_407 = None
        mul_408: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_127, primals_80);  primals_80 = None
        mul_409: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_127, sub_14);  div_127 = None
        sum_128: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1], True);  mul_409 = None
        view_501: "f32[768]" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
        neg_36: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_408)
        sum_129: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_36, [2], True);  neg_36 = None
        add_159: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_395, mul_408);  mul_395 = mul_408 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_410: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_9, 2)
        div_128: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_127, mul_410);  sum_127 = mul_410 = None
        eq_26: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_9, 0);  sqrt_9 = None
        where_33: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_26, full_default_13, div_128);  eq_26 = div_128 = None
        mul_411: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_33, 0.002607561929595828);  where_33 = None
        mul_412: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_411, sub_14);  mul_411 = sub_14 = None
        add_160: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_159, mul_412);  add_159 = mul_412 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_62: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_129, [128, 128, 768]);  sum_129 = None
        div_129: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_62, 768);  expand_62 = None
        add_161: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_160, div_129);  add_160 = div_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_38: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_413: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_414: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_413);  mul_413 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_502: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_414, [16384, 768]);  mul_414 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_92: "f32[16384, 768]" = torch.ops.aten.mm.default(view_502, permute_381);  permute_381 = None
        permute_382: "f32[768, 16384]" = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
        sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_502, [0], True);  view_502 = None
        view_503: "f32[768]" = torch.ops.aten.reshape.default(sum_130, [768]);  sum_130 = None
        view_504: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_92, [128, 128, 768]);  mm_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_505: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_504, [128, 128, 12, 64]);  view_504 = None
        permute_385: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_115: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_506: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_115, [1536, 128, 64]);  clone_115 = None
        bmm_52: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_386, view_506);  permute_386 = None
        bmm_53: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_506, permute_387);  view_506 = permute_387 = None
        view_507: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_52, [128, 12, 128, 64]);  bmm_52 = None
        view_508: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_53, [128, 12, 128, 128]);  bmm_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_39: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_415: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_416: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_508, mul_415);  view_508 = mul_415 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_417: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_416, div_18);  mul_416 = None
        sum_131: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [-1], True)
        neg_37: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_7: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_37, sum_131, mul_417);  neg_37 = sum_131 = mul_417 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_34: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_7);  fma_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_130: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_34, 8.0);  where_34 = None
        view_509: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_130, [1536, 128, 128]);  div_130 = None
        bmm_54: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_388, view_509);  permute_388 = None
        bmm_55: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_509, permute_389);  view_509 = permute_389 = None
        view_510: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_54, [128, 12, 64, 128]);  bmm_54 = None
        view_511: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_55, [128, 12, 128, 64]);  bmm_55 = None
        permute_390: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_510, [0, 1, 3, 2]);  view_510 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_391: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_117: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_512: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_117, [128, 128, 768]);  clone_117 = None
        view_513: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_512, [16384, 768]);  view_512 = None
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_392: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_94: "f32[16384, 768]" = torch.ops.aten.mm.default(view_513, permute_392);  permute_392 = None
        permute_393: "f32[768, 16384]" = torch.ops.aten.permute.default(view_513, [1, 0])
        mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_393, view_88);  permute_393 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
        view_514: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        view_515: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_94, [128, 128, 768]);  mm_94 = None
        permute_396: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
        view_516: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_396, [128, 128, 768]);  permute_396 = None
        clone_118: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_516, memory_format = torch.contiguous_format);  view_516 = None
        view_517: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_118, [16384, 768]);  clone_118 = None
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_96: "f32[16384, 768]" = torch.ops.aten.mm.default(view_517, permute_397);  permute_397 = None
        permute_398: "f32[768, 16384]" = torch.ops.aten.permute.default(view_517, [1, 0])
        mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = None
        sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
        view_518: "f32[768]" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
        view_519: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_96, [128, 128, 768]);  mm_96 = None
        add_162: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_515, view_519);  view_515 = view_519 = None
        permute_401: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        clone_119: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_520: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_119, [128, 128, 768]);  clone_119 = None
        view_521: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_520, [16384, 768]);  view_520 = None
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_98: "f32[16384, 768]" = torch.ops.aten.mm.default(view_521, permute_402);  permute_402 = None
        permute_403: "f32[768, 16384]" = torch.ops.aten.permute.default(view_521, [1, 0])
        mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
        sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
        view_522: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
        view_523: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_98, [128, 128, 768]);  mm_98 = None
        add_163: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_162, view_523);  add_162 = view_523 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_135: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1], True)
        view_524: "f32[768]" = torch.ops.aten.reshape.default(sum_135, [768]);  sum_135 = None
        mul_62: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_70, sub_12)
        add_30: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_8, 1e-06)
        div_16: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_62, add_30);  mul_62 = None
        div_132: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_16, add_30);  div_16 = None
        neg_38: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_163)
        mul_418: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_38, div_132);  neg_38 = div_132 = None
        div_133: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_163, add_30);  add_163 = add_30 = None
        sum_136: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_133, primals_70);  primals_70 = None
        mul_420: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_133, sub_12);  div_133 = None
        sum_137: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1], True);  mul_420 = None
        view_525: "f32[768]" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
        neg_39: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_419)
        sum_138: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True);  neg_39 = None
        add_164: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_161, mul_419);  add_161 = mul_419 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_421: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_8, 2)
        div_134: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_136, mul_421);  sum_136 = mul_421 = None
        eq_27: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_8, 0);  sqrt_8 = None
        where_35: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_27, full_default_13, div_134);  eq_27 = div_134 = None
        mul_422: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_35, 0.002607561929595828);  where_35 = None
        mul_423: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_422, sub_12);  mul_422 = sub_12 = None
        add_165: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_164, mul_423);  add_164 = mul_423 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_63: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_138, [128, 128, 768]);  sum_138 = None
        div_135: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_63, 768);  expand_63 = None
        add_166: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_165, div_135);  add_165 = div_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_40: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_424: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_425: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_166, mul_424);  add_166 = mul_424 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_41: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_426: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_427: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul_426);  mul_426 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_526: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_427, [16384, 768]);  mul_427 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_100: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_526, permute_406);  permute_406 = None
        permute_407: "f32[768, 16384]" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
        sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
        view_527: "f32[768]" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
        view_528: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_100, [128, 128, 3072]);  mm_100 = None
        convert_element_type_42: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_428: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_429: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_528, mul_428);  view_528 = mul_428 = None
        view_85: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_22, [128, 128, 3072]);  addmm_22 = None
        mul_54: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_28: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_431: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
        mul_432: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_433: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_432, -0.5);  mul_432 = None
        exp_24: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_433);  mul_433 = None
        mul_434: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_435: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_434);  view_85 = mul_434 = None
        add_168: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_431, mul_435);  mul_431 = mul_435 = None
        mul_436: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_429, add_168);  mul_429 = add_168 = None
        view_529: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_436, [16384, 3072]);  mul_436 = None
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_410: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_102: "f32[16384, 768]" = torch.ops.aten.mm.default(view_529, permute_410);  permute_410 = None
        permute_411: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
        sum_140: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        view_530: "f32[3072]" = torch.ops.aten.reshape.default(sum_140, [3072]);  sum_140 = None
        view_531: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_102, [128, 128, 768]);  mm_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_141: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_531, [0, 1], True)
        view_532: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        mul_52: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_64, sub_11)
        add_26: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_7, 1e-06)
        div_15: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_52, add_26);  mul_52 = None
        div_137: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_15, add_26);  div_15 = None
        neg_40: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_531)
        mul_437: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_40, div_137);  neg_40 = div_137 = None
        div_138: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_531, add_26);  view_531 = add_26 = None
        sum_142: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True);  mul_437 = None
        mul_438: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_138, primals_64);  primals_64 = None
        mul_439: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_138, sub_11);  div_138 = None
        sum_143: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1], True);  mul_439 = None
        view_533: "f32[768]" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
        neg_41: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_438)
        sum_144: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_41, [2], True);  neg_41 = None
        add_169: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_425, mul_438);  mul_425 = mul_438 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_440: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_7, 2)
        div_139: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_142, mul_440);  sum_142 = mul_440 = None
        eq_28: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_7, 0);  sqrt_7 = None
        where_36: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_28, full_default_13, div_139);  eq_28 = div_139 = None
        mul_441: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_36, 0.002607561929595828);  where_36 = None
        mul_442: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_441, sub_11);  mul_441 = sub_11 = None
        add_170: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_169, mul_442);  add_169 = mul_442 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_64: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_144, [128, 128, 768]);  sum_144 = None
        div_140: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_64, 768);  expand_64 = None
        add_171: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_170, div_140);  add_170 = div_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_43: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_443: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_444: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_443);  mul_443 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_534: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_444, [16384, 768]);  mul_444 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_104: "f32[16384, 768]" = torch.ops.aten.mm.default(view_534, permute_414);  permute_414 = None
        permute_415: "f32[768, 16384]" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
        sum_145: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        view_535: "f32[768]" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
        view_536: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_104, [128, 128, 768]);  mm_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_537: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_536, [128, 128, 12, 64]);  view_536 = None
        permute_418: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_124: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_538: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_124, [1536, 128, 64]);  clone_124 = None
        bmm_56: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_419, view_538);  permute_419 = None
        bmm_57: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_538, permute_420);  view_538 = permute_420 = None
        view_539: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_56, [128, 12, 128, 64]);  bmm_56 = None
        view_540: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_57, [128, 12, 128, 128]);  bmm_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_44: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_445: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_446: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_540, mul_445);  view_540 = mul_445 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_447: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_446, div_14);  mul_446 = None
        sum_146: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [-1], True)
        neg_42: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_8: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_42, sum_146, mul_447);  neg_42 = sum_146 = mul_447 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_37: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_8);  fma_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_141: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_37, 8.0);  where_37 = None
        view_541: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_141, [1536, 128, 128]);  div_141 = None
        bmm_58: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_421, view_541);  permute_421 = None
        bmm_59: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_541, permute_422);  view_541 = permute_422 = None
        view_542: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_58, [128, 12, 64, 128]);  bmm_58 = None
        view_543: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_59, [128, 12, 128, 64]);  bmm_59 = None
        permute_423: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 2]);  view_542 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_424: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
        clone_126: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_544: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_126, [128, 128, 768]);  clone_126 = None
        view_545: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_544, [16384, 768]);  view_544 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_425: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_106: "f32[16384, 768]" = torch.ops.aten.mm.default(view_545, permute_425);  permute_425 = None
        permute_426: "f32[768, 16384]" = torch.ops.aten.permute.default(view_545, [1, 0])
        mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_426, view_66);  permute_426 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
        view_546: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_547: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_106, [128, 128, 768]);  mm_106 = None
        permute_429: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
        view_548: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_429, [128, 128, 768]);  permute_429 = None
        clone_127: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_548, memory_format = torch.contiguous_format);  view_548 = None
        view_549: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_127, [16384, 768]);  clone_127 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_108: "f32[16384, 768]" = torch.ops.aten.mm.default(view_549, permute_430);  permute_430 = None
        permute_431: "f32[768, 16384]" = torch.ops.aten.permute.default(view_549, [1, 0])
        mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = None
        sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
        view_550: "f32[768]" = torch.ops.aten.reshape.default(sum_148, [768]);  sum_148 = None
        view_551: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_108, [128, 128, 768]);  mm_108 = None
        add_172: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_547, view_551);  view_547 = view_551 = None
        permute_434: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
        clone_128: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_552: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_128, [128, 128, 768]);  clone_128 = None
        view_553: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_552, [16384, 768]);  view_552 = None
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_110: "f32[16384, 768]" = torch.ops.aten.mm.default(view_553, permute_435);  permute_435 = None
        permute_436: "f32[768, 16384]" = torch.ops.aten.permute.default(view_553, [1, 0])
        mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
        sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
        view_554: "f32[768]" = torch.ops.aten.reshape.default(sum_149, [768]);  sum_149 = None
        view_555: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_110, [128, 128, 768]);  mm_110 = None
        add_173: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_172, view_555);  add_172 = view_555 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_150: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1], True)
        view_556: "f32[768]" = torch.ops.aten.reshape.default(sum_150, [768]);  sum_150 = None
        mul_47: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_54, sub_9)
        add_23: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_6, 1e-06)
        div_12: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_47, add_23);  mul_47 = None
        div_143: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_12, add_23);  div_12 = None
        neg_43: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_173)
        mul_448: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_43, div_143);  neg_43 = div_143 = None
        div_144: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_173, add_23);  add_173 = add_23 = None
        sum_151: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_144, primals_54);  primals_54 = None
        mul_450: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_144, sub_9);  div_144 = None
        sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1], True);  mul_450 = None
        view_557: "f32[768]" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
        neg_44: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_449)
        sum_153: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_44, [2], True);  neg_44 = None
        add_174: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_171, mul_449);  add_171 = mul_449 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_451: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_6, 2)
        div_145: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_151, mul_451);  sum_151 = mul_451 = None
        eq_29: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_6, 0);  sqrt_6 = None
        where_38: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_29, full_default_13, div_145);  eq_29 = div_145 = None
        mul_452: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_38, 0.002607561929595828);  where_38 = None
        mul_453: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_452, sub_9);  mul_452 = sub_9 = None
        add_175: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_174, mul_453);  add_174 = mul_453 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_65: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_153, [128, 128, 768]);  sum_153 = None
        div_146: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_65, 768);  expand_65 = None
        add_176: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_175, div_146);  add_175 = div_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_45: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_454: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_455: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_454);  add_176 = mul_454 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_46: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_456: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_457: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_455, mul_456);  mul_456 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_558: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_457, [16384, 768]);  mul_457 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_112: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_558, permute_439);  permute_439 = None
        permute_440: "f32[768, 16384]" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
        sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
        view_559: "f32[768]" = torch.ops.aten.reshape.default(sum_154, [768]);  sum_154 = None
        view_560: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_112, [128, 128, 3072]);  mm_112 = None
        convert_element_type_47: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_458: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_459: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_560, mul_458);  view_560 = mul_458 = None
        view_63: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_16, [128, 128, 3072]);  addmm_16 = None
        mul_39: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_21: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_461: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_462: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_463: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_462, -0.5);  mul_462 = None
        exp_25: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_463);  mul_463 = None
        mul_464: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_465: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_464);  view_63 = mul_464 = None
        add_178: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_461, mul_465);  mul_461 = mul_465 = None
        mul_466: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_459, add_178);  mul_459 = add_178 = None
        view_561: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_466, [16384, 3072]);  mul_466 = None
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_443: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_114: "f32[16384, 768]" = torch.ops.aten.mm.default(view_561, permute_443);  permute_443 = None
        permute_444: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_561, [1, 0])
        mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
        sum_155: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
        view_562: "f32[3072]" = torch.ops.aten.reshape.default(sum_155, [3072]);  sum_155 = None
        view_563: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_114, [128, 128, 768]);  mm_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_156: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1], True)
        view_564: "f32[768]" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
        mul_37: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_48, sub_8)
        add_19: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_5, 1e-06)
        div_11: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_37, add_19);  mul_37 = None
        div_148: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_11, add_19);  div_11 = None
        neg_45: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_563)
        mul_467: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_45, div_148);  neg_45 = div_148 = None
        div_149: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_563, add_19);  view_563 = add_19 = None
        sum_157: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True);  mul_467 = None
        mul_468: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_149, primals_48);  primals_48 = None
        mul_469: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_149, sub_8);  div_149 = None
        sum_158: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1], True);  mul_469 = None
        view_565: "f32[768]" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
        neg_46: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_468)
        sum_159: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_46, [2], True);  neg_46 = None
        add_179: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_455, mul_468);  mul_455 = mul_468 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_470: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_5, 2)
        div_150: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_157, mul_470);  sum_157 = mul_470 = None
        eq_30: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_5, 0);  sqrt_5 = None
        where_39: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_30, full_default_13, div_150);  eq_30 = div_150 = None
        mul_471: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_39, 0.002607561929595828);  where_39 = None
        mul_472: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_471, sub_8);  mul_471 = sub_8 = None
        add_180: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_179, mul_472);  add_179 = mul_472 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_66: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_159, [128, 128, 768]);  sum_159 = None
        div_151: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_66, 768);  expand_66 = None
        add_181: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_180, div_151);  add_180 = div_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_48: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_473: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_474: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_181, mul_473);  mul_473 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_566: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_474, [16384, 768]);  mul_474 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_447: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_116: "f32[16384, 768]" = torch.ops.aten.mm.default(view_566, permute_447);  permute_447 = None
        permute_448: "f32[768, 16384]" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
        sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        view_567: "f32[768]" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
        view_568: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_116, [128, 128, 768]);  mm_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_569: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_568, [128, 128, 12, 64]);  view_568 = None
        permute_451: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_133: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_570: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_133, [1536, 128, 64]);  clone_133 = None
        bmm_60: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_452, view_570);  permute_452 = None
        bmm_61: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_570, permute_453);  view_570 = permute_453 = None
        view_571: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_60, [128, 12, 128, 64]);  bmm_60 = None
        view_572: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_61, [128, 12, 128, 128]);  bmm_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_49: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_475: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_476: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_572, mul_475);  view_572 = mul_475 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_477: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_476, div_10);  mul_476 = None
        sum_161: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [-1], True)
        neg_47: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_9: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_47, sum_161, mul_477);  neg_47 = sum_161 = mul_477 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_40: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_9);  fma_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_152: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_40, 8.0);  where_40 = None
        view_573: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_152, [1536, 128, 128]);  div_152 = None
        bmm_62: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_454, view_573);  permute_454 = None
        bmm_63: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_573, permute_455);  view_573 = permute_455 = None
        view_574: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_62, [128, 12, 64, 128]);  bmm_62 = None
        view_575: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_63, [128, 12, 128, 64]);  bmm_63 = None
        permute_456: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2]);  view_574 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_457: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
        clone_135: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_576: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_135, [128, 128, 768]);  clone_135 = None
        view_577: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_576, [16384, 768]);  view_576 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_458: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_118: "f32[16384, 768]" = torch.ops.aten.mm.default(view_577, permute_458);  permute_458 = None
        permute_459: "f32[768, 16384]" = torch.ops.aten.permute.default(view_577, [1, 0])
        mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_459, view_44);  permute_459 = None
        sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
        view_578: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        view_579: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_118, [128, 128, 768]);  mm_118 = None
        permute_462: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
        view_580: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_462, [128, 128, 768]);  permute_462 = None
        clone_136: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_580, memory_format = torch.contiguous_format);  view_580 = None
        view_581: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_136, [16384, 768]);  clone_136 = None
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_120: "f32[16384, 768]" = torch.ops.aten.mm.default(view_581, permute_463);  permute_463 = None
        permute_464: "f32[768, 16384]" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = None
        sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
        view_582: "f32[768]" = torch.ops.aten.reshape.default(sum_163, [768]);  sum_163 = None
        view_583: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_120, [128, 128, 768]);  mm_120 = None
        add_182: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_579, view_583);  view_579 = view_583 = None
        permute_467: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_575, [0, 2, 1, 3]);  view_575 = None
        clone_137: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_584: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_137, [128, 128, 768]);  clone_137 = None
        view_585: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_584, [16384, 768]);  view_584 = None
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_122: "f32[16384, 768]" = torch.ops.aten.mm.default(view_585, permute_468);  permute_468 = None
        permute_469: "f32[768, 16384]" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
        sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
        view_586: "f32[768]" = torch.ops.aten.reshape.default(sum_164, [768]);  sum_164 = None
        view_587: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_122, [128, 128, 768]);  mm_122 = None
        add_183: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_182, view_587);  add_182 = view_587 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_165: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 1], True)
        view_588: "f32[768]" = torch.ops.aten.reshape.default(sum_165, [768]);  sum_165 = None
        mul_32: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_38, sub_6)
        add_16: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-06)
        div_8: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_32, add_16);  mul_32 = None
        div_154: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_8, add_16);  div_8 = None
        neg_48: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_183)
        mul_478: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_48, div_154);  neg_48 = div_154 = None
        div_155: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_183, add_16);  add_183 = add_16 = None
        sum_166: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_155, primals_38);  primals_38 = None
        mul_480: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_155, sub_6);  div_155 = None
        sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 1], True);  mul_480 = None
        view_589: "f32[768]" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
        neg_49: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_479)
        sum_168: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_49, [2], True);  neg_49 = None
        add_184: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_181, mul_479);  add_181 = mul_479 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_481: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_4, 2)
        div_156: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_166, mul_481);  sum_166 = mul_481 = None
        eq_31: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_4, 0);  sqrt_4 = None
        where_41: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_31, full_default_13, div_156);  eq_31 = div_156 = None
        mul_482: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_41, 0.002607561929595828);  where_41 = None
        mul_483: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_482, sub_6);  mul_482 = sub_6 = None
        add_185: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_184, mul_483);  add_184 = mul_483 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_67: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_168, [128, 128, 768]);  sum_168 = None
        div_157: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_67, 768);  expand_67 = None
        add_186: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_185, div_157);  add_185 = div_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_50: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_484: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_485: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_186, mul_484);  add_186 = mul_484 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_51: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_486: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_487: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_486 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_590: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_487, [16384, 768]);  mul_487 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_472: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_124: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_590, permute_472);  permute_472 = None
        permute_473: "f32[768, 16384]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
        sum_169: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f32[768]" = torch.ops.aten.reshape.default(sum_169, [768]);  sum_169 = None
        view_592: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_124, [128, 128, 3072]);  mm_124 = None
        convert_element_type_52: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_488: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_489: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_592, mul_488);  view_592 = mul_488 = None
        view_41: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_10, [128, 128, 3072]);  addmm_10 = None
        mul_24: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_14: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_491: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_492: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_493: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_492, -0.5);  mul_492 = None
        exp_26: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_493);  mul_493 = None
        mul_494: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_495: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_494);  view_41 = mul_494 = None
        add_188: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_491, mul_495);  mul_491 = mul_495 = None
        mul_496: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_489, add_188);  mul_489 = add_188 = None
        view_593: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_496, [16384, 3072]);  mul_496 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_476: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_126: "f32[16384, 768]" = torch.ops.aten.mm.default(view_593, permute_476);  permute_476 = None
        permute_477: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
        sum_170: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        view_594: "f32[3072]" = torch.ops.aten.reshape.default(sum_170, [3072]);  sum_170 = None
        view_595: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_126, [128, 128, 768]);  mm_126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_171: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0, 1], True)
        view_596: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
        mul_22: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_32, sub_5)
        add_12: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-06)
        div_7: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_22, add_12);  mul_22 = None
        div_159: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_7, add_12);  div_7 = None
        neg_50: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_595)
        mul_497: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_50, div_159);  neg_50 = div_159 = None
        div_160: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_595, add_12);  view_595 = add_12 = None
        sum_172: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_160, primals_32);  primals_32 = None
        mul_499: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_160, sub_5);  div_160 = None
        sum_173: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 1], True);  mul_499 = None
        view_597: "f32[768]" = torch.ops.aten.reshape.default(sum_173, [768]);  sum_173 = None
        neg_51: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_498)
        sum_174: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_51, [2], True);  neg_51 = None
        add_189: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_485, mul_498);  mul_485 = mul_498 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_500: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_3, 2)
        div_161: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_172, mul_500);  sum_172 = mul_500 = None
        eq_32: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_3, 0);  sqrt_3 = None
        where_42: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_32, full_default_13, div_161);  eq_32 = div_161 = None
        mul_501: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_42, 0.002607561929595828);  where_42 = None
        mul_502: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_501, sub_5);  mul_501 = sub_5 = None
        add_190: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_189, mul_502);  add_189 = mul_502 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_68: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_174, [128, 128, 768]);  sum_174 = None
        div_162: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_68, 768);  expand_68 = None
        add_191: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_190, div_162);  add_190 = div_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_53: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_503: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_504: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_191, mul_503);  mul_503 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_598: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_504, [16384, 768]);  mul_504 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_128: "f32[16384, 768]" = torch.ops.aten.mm.default(view_598, permute_480);  permute_480 = None
        permute_481: "f32[768, 16384]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
        sum_175: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
        view_599: "f32[768]" = torch.ops.aten.reshape.default(sum_175, [768]);  sum_175 = None
        view_600: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_128, [128, 128, 768]);  mm_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_601: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_600, [128, 128, 12, 64]);  view_600 = None
        permute_484: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_142: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_602: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_142, [1536, 128, 64]);  clone_142 = None
        bmm_64: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_485, view_602);  permute_485 = None
        bmm_65: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_602, permute_486);  view_602 = permute_486 = None
        view_603: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_64, [128, 12, 128, 64]);  bmm_64 = None
        view_604: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_65, [128, 12, 128, 128]);  bmm_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_54: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_505: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_506: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_604, mul_505);  view_604 = mul_505 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_507: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_506, div_6);  mul_506 = None
        sum_176: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [-1], True)
        neg_52: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_10: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_52, sum_176, mul_507);  neg_52 = sum_176 = mul_507 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_43: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_10);  fma_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_163: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_43, 8.0);  where_43 = None
        view_605: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_163, [1536, 128, 128]);  div_163 = None
        bmm_66: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_487, view_605);  permute_487 = None
        bmm_67: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_605, permute_488);  view_605 = permute_488 = None
        view_606: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_66, [128, 12, 64, 128]);  bmm_66 = None
        view_607: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_67, [128, 12, 128, 64]);  bmm_67 = None
        permute_489: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2]);  view_606 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_490: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
        clone_144: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_608: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_144, [128, 128, 768]);  clone_144 = None
        view_609: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_608, [16384, 768]);  view_608 = None
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_130: "f32[16384, 768]" = torch.ops.aten.mm.default(view_609, permute_491);  permute_491 = None
        permute_492: "f32[768, 16384]" = torch.ops.aten.permute.default(view_609, [1, 0])
        mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_492, view_22);  permute_492 = None
        sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
        view_610: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        view_611: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_130, [128, 128, 768]);  mm_130 = None
        permute_495: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        view_612: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_495, [128, 128, 768]);  permute_495 = None
        clone_145: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_612, memory_format = torch.contiguous_format);  view_612 = None
        view_613: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_145, [16384, 768]);  clone_145 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_132: "f32[16384, 768]" = torch.ops.aten.mm.default(view_613, permute_496);  permute_496 = None
        permute_497: "f32[768, 16384]" = torch.ops.aten.permute.default(view_613, [1, 0])
        mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = None
        sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_613, [0], True);  view_613 = None
        view_614: "f32[768]" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
        view_615: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_132, [128, 128, 768]);  mm_132 = None
        add_192: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_611, view_615);  view_611 = view_615 = None
        permute_500: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
        clone_146: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_616: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_146, [128, 128, 768]);  clone_146 = None
        view_617: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_616, [16384, 768]);  view_616 = None
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_134: "f32[16384, 768]" = torch.ops.aten.mm.default(view_617, permute_501);  permute_501 = None
        permute_502: "f32[768, 16384]" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
        sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
        view_618: "f32[768]" = torch.ops.aten.reshape.default(sum_179, [768]);  sum_179 = None
        view_619: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_134, [128, 128, 768]);  mm_134 = None
        add_193: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_192, view_619);  add_192 = view_619 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_180: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1], True)
        view_620: "f32[768]" = torch.ops.aten.reshape.default(sum_180, [768]);  sum_180 = None
        mul_17: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_22, sub_3)
        add_9: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06)
        div_4: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_17, add_9);  mul_17 = None
        div_165: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_4, add_9);  div_4 = None
        neg_53: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_193)
        mul_508: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_53, div_165);  neg_53 = div_165 = None
        div_166: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_193, add_9);  add_193 = add_9 = None
        sum_181: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_166, primals_22);  primals_22 = None
        mul_510: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_166, sub_3);  div_166 = None
        sum_182: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1], True);  mul_510 = None
        view_621: "f32[768]" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
        neg_54: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_509)
        sum_183: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_54, [2], True);  neg_54 = None
        add_194: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_191, mul_509);  add_191 = mul_509 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_511: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_2, 2)
        div_167: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_181, mul_511);  sum_181 = mul_511 = None
        eq_33: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_2, 0);  sqrt_2 = None
        where_44: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_33, full_default_13, div_167);  eq_33 = div_167 = None
        mul_512: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_44, 0.002607561929595828);  where_44 = None
        mul_513: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_512, sub_3);  mul_512 = sub_3 = None
        add_195: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_194, mul_513);  add_194 = mul_513 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_69: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_183, [128, 128, 768]);  sum_183 = None
        div_168: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_69, 768);  expand_69 = None
        add_196: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_195, div_168);  add_195 = div_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_55: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_514: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_515: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_196, mul_514);  add_196 = mul_514 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_56: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_516: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_517: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_515, mul_516);  mul_516 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_622: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_517, [16384, 768]);  mul_517 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_505: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_136: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_622, permute_505);  permute_505 = None
        permute_506: "f32[768, 16384]" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
        sum_184: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        view_623: "f32[768]" = torch.ops.aten.reshape.default(sum_184, [768]);  sum_184 = None
        view_624: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_136, [128, 128, 3072]);  mm_136 = None
        convert_element_type_57: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_518: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_519: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_624, mul_518);  view_624 = mul_518 = None
        view_19: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, [128, 128, 3072]);  addmm_4 = None
        mul_9: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_521: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_522: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_523: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_522, -0.5);  mul_522 = None
        exp_27: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_523);  mul_523 = None
        mul_524: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_525: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_524);  view_19 = mul_524 = None
        add_198: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_521, mul_525);  mul_521 = mul_525 = None
        mul_526: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_519, add_198);  mul_519 = add_198 = None
        view_625: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_526, [16384, 3072]);  mul_526 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_509: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_138: "f32[16384, 768]" = torch.ops.aten.mm.default(view_625, permute_509);  permute_509 = None
        permute_510: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
        sum_185: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        view_626: "f32[3072]" = torch.ops.aten.reshape.default(sum_185, [3072]);  sum_185 = None
        view_627: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_138, [128, 128, 768]);  mm_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_186: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_627, [0, 1], True)
        view_628: "f32[768]" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
        mul_7: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_16, sub_2)
        add_5: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-06)
        div_3: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_7, add_5);  mul_7 = None
        div_170: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_3, add_5);  div_3 = None
        neg_55: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_627)
        mul_527: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_55, div_170);  neg_55 = div_170 = None
        div_171: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_627, add_5);  view_627 = add_5 = None
        sum_187: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True);  mul_527 = None
        mul_528: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_171, primals_16);  primals_16 = None
        mul_529: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_171, sub_2);  div_171 = None
        sum_188: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 1], True);  mul_529 = None
        view_629: "f32[768]" = torch.ops.aten.reshape.default(sum_188, [768]);  sum_188 = None
        neg_56: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_528)
        sum_189: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_56, [2], True);  neg_56 = None
        add_199: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_515, mul_528);  mul_515 = mul_528 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_530: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_1, 2)
        div_172: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_187, mul_530);  sum_187 = mul_530 = None
        eq_34: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_1, 0);  sqrt_1 = None
        where_45: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_34, full_default_13, div_172);  eq_34 = div_172 = None
        mul_531: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_45, 0.002607561929595828);  where_45 = None
        mul_532: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_531, sub_2);  mul_531 = sub_2 = None
        add_200: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_199, mul_532);  add_199 = mul_532 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_70: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_189, [128, 128, 768]);  sum_189 = None
        div_173: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_70, 768);  expand_70 = None
        add_201: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_200, div_173);  add_200 = div_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_58: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_533: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_534: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_201, mul_533);  mul_533 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_630: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_534, [16384, 768]);  mul_534 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_140: "f32[16384, 768]" = torch.ops.aten.mm.default(view_630, permute_513);  permute_513 = None
        permute_514: "f32[768, 16384]" = torch.ops.aten.permute.default(view_630, [1, 0])
        mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
        sum_190: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_630, [0], True);  view_630 = None
        view_631: "f32[768]" = torch.ops.aten.reshape.default(sum_190, [768]);  sum_190 = None
        view_632: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_140, [128, 128, 768]);  mm_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_633: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_632, [128, 128, 12, 64]);  view_632 = None
        permute_517: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_151: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_634: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_151, [1536, 128, 64]);  clone_151 = None
        bmm_68: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(permute_518, view_634);  permute_518 = None
        bmm_69: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_634, permute_519);  view_634 = permute_519 = None
        view_635: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_68, [128, 12, 128, 64]);  bmm_68 = None
        view_636: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, [128, 12, 128, 128]);  bmm_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_59: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_535: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_536: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_636, mul_535);  view_636 = mul_535 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        view_11: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, [128, 12, 128, 128]);  bmm = None
        div_1: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        full_default: "f32[]" = torch.ops.aten.full.default([], -1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_1);  full_default = div_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        sub_1: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_2: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_537: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_536, div_2);  mul_536 = None
        sum_191: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [-1], True)
        neg_57: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_11: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_57, sum_191, mul_537);  neg_57 = sum_191 = mul_537 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_46: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_11);  eq = fma_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_174: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_46, 8.0);  where_46 = None
        view_637: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_174, [1536, 128, 128]);  div_174 = None
        bmm_70: "f32[1536, 64, 128]" = torch.ops.aten.bmm.default(permute_520, view_637);  permute_520 = None
        bmm_71: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_637, permute_521);  view_637 = permute_521 = None
        view_638: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_70, [128, 12, 64, 128]);  bmm_70 = None
        view_639: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_71, [128, 12, 128, 64]);  bmm_71 = None
        permute_522: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_638, [0, 1, 3, 2]);  view_638 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_523: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
        clone_153: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_640: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_153, [128, 128, 768]);  clone_153 = None
        view_641: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_640, [16384, 768]);  view_640 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_524: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_142: "f32[16384, 768]" = torch.ops.aten.mm.default(view_641, permute_524);  permute_524 = None
        permute_525: "f32[768, 16384]" = torch.ops.aten.permute.default(view_641, [1, 0])
        mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_525, view);  permute_525 = None
        sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_641, [0], True);  view_641 = None
        view_642: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        view_643: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_142, [128, 128, 768]);  mm_142 = None
        permute_528: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        view_644: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_528, [128, 128, 768]);  permute_528 = None
        clone_154: "f32[128, 128, 768]" = torch.ops.aten.clone.default(view_644, memory_format = torch.contiguous_format);  view_644 = None
        view_645: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_154, [16384, 768]);  clone_154 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_144: "f32[16384, 768]" = torch.ops.aten.mm.default(view_645, permute_529);  permute_529 = None
        permute_530: "f32[768, 16384]" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = None
        sum_193: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
        view_646: "f32[768]" = torch.ops.aten.reshape.default(sum_193, [768]);  sum_193 = None
        view_647: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_144, [128, 128, 768]);  mm_144 = None
        add_202: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_643, view_647);  view_643 = view_647 = None
        permute_533: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_639, [0, 2, 1, 3]);  view_639 = None
        clone_155: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_648: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_155, [128, 128, 768]);  clone_155 = None
        view_649: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_648, [16384, 768]);  view_648 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_146: "f32[16384, 768]" = torch.ops.aten.mm.default(view_649, permute_534);  permute_534 = None
        permute_535: "f32[768, 16384]" = torch.ops.aten.permute.default(view_649, [1, 0])
        mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
        sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_649, [0], True);  view_649 = None
        view_650: "f32[768]" = torch.ops.aten.reshape.default(sum_194, [768]);  sum_194 = None
        view_651: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_146, [128, 128, 768]);  mm_146 = None
        add_203: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_202, view_651);  add_202 = view_651 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_195: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1], True)
        view_652: "f32[768]" = torch.ops.aten.reshape.default(sum_195, [768]);  sum_195 = None
        mul_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_6, sub)
        add_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_2, add_2);  mul_2 = None
        div_176: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div, add_2);  div = None
        neg_58: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_203)
        mul_538: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_58, div_176);  neg_58 = div_176 = None
        div_177: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_203, add_2);  add_203 = add_2 = None
        sum_196: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_177, primals_6);  primals_6 = None
        mul_540: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_177, sub);  div_177 = None
        sum_197: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 1], True);  mul_540 = None
        view_653: "f32[768]" = torch.ops.aten.reshape.default(sum_197, [768]);  sum_197 = None
        neg_59: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_539)
        sum_198: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_59, [2], True);  neg_59 = None
        add_204: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_201, mul_539);  add_201 = mul_539 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_541: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt, 2)
        div_178: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_196, mul_541);  sum_196 = mul_541 = None
        eq_35: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt, 0);  sqrt = None
        where_47: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_35, full_default_13, div_178);  eq_35 = div_178 = None
        mul_542: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_47, 0.002607561929595828);  where_47 = None
        mul_543: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_542, sub);  mul_542 = sub = None
        add_205: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_204, mul_543);  add_204 = mul_543 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_71: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_198, [128, 128, 768]);  sum_198 = None
        div_179: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_71, 768);  expand_71 = None
        add_206: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_205, div_179);  add_205 = div_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        convert_element_type_60: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_544: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_545: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_206, mul_544);  add_206 = mul_544 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        eq_36: "b8[128, 128]" = torch.ops.aten.eq.Scalar(primals_5, 0)
        unsqueeze_2: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_36, -1);  eq_36 = None
        where_48: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_13, mul_545);  unsqueeze_2 = None
        full_default_50: "f32[3, 768]" = torch.ops.aten.full.default([3, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[3, 768]" = torch.ops.aten.index_put.default(full_default_50, [primals_5], where_48, True);  full_default_50 = primals_5 = where_48 = None
        eq_37: "b8[128, 128]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_3: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_37, -1);  eq_37 = None
        where_49: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_13, mul_545);  unsqueeze_3 = full_default_13 = mul_545 = None
        full_default_52: "f32[20005, 768]" = torch.ops.aten.full.default([20005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[20005, 768]" = torch.ops.aten.index_put.default(full_default_52, [primals_1], where_49, True);  full_default_52 = primals_1 = where_49 = None
        return (None, index_put_1, None, index_put, None, view_653, view_652, mm_147, view_650, mm_145, view_646, mm_143, view_642, mm_141, view_631, view_629, view_628, mm_139, view_626, mm_137, view_623, view_621, view_620, mm_135, view_618, mm_133, view_614, mm_131, view_610, mm_129, view_599, view_597, view_596, mm_127, view_594, mm_125, view_591, view_589, view_588, mm_123, view_586, mm_121, view_582, mm_119, view_578, mm_117, view_567, view_565, view_564, mm_115, view_562, mm_113, view_559, view_557, view_556, mm_111, view_554, mm_109, view_550, mm_107, view_546, mm_105, view_535, view_533, view_532, mm_103, view_530, mm_101, view_527, view_525, view_524, mm_99, view_522, mm_97, view_518, mm_95, view_514, mm_93, view_503, view_501, view_500, mm_91, view_498, mm_89, view_495, view_493, view_492, mm_87, view_490, mm_85, view_486, mm_83, view_482, mm_81, view_471, view_469, view_468, mm_79, view_466, mm_77, view_463, view_461, view_460, mm_75, view_458, mm_73, view_454, mm_71, view_450, mm_69, view_439, view_437, view_436, mm_67, view_434, mm_65, view_431, view_429, view_428, mm_63, view_426, mm_61, view_422, mm_59, view_418, mm_57, view_407, view_405, view_404, mm_55, view_402, mm_53, view_399, view_397, view_396, mm_51, view_394, mm_49, view_390, mm_47, view_386, mm_45, view_375, view_373, view_372, mm_43, view_370, mm_41, view_367, view_365, view_364, mm_39, view_362, mm_37, view_358, mm_35, view_354, mm_33, view_343, view_341, view_340, mm_31, view_338, mm_29, view_335, view_333, view_332, mm_27, view_330, mm_25, view_326, mm_23, view_322, mm_21, view_311, view_309, view_308, mm_19, view_306, mm_17, view_303, view_301, view_300, mm_15, view_298, mm_13, view_294, mm_11, view_290, mm_9, view_279, view_277, view_276, mm_7, view_274, mm_5, view_271, mm_3, view_269, mm_1, view_267)
