class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 64]", primals_2: "f32[50304, 768]", primals_4: "f32[768]", primals_6: "f32[2304, 768]", primals_8: "f32[768, 768]", primals_10: "f32[768]", primals_12: "f32[3072, 768]", primals_14: "f32[768, 3072]", primals_16: "f32[768]", primals_18: "f32[2304, 768]", primals_20: "f32[768, 768]", primals_22: "f32[768]", primals_24: "f32[3072, 768]", primals_26: "f32[768, 3072]", primals_28: "f32[768]", primals_30: "f32[2304, 768]", primals_32: "f32[768, 768]", primals_34: "f32[768]", primals_36: "f32[3072, 768]", primals_38: "f32[768, 3072]", primals_40: "f32[768]", primals_42: "f32[2304, 768]", primals_44: "f32[768, 768]", primals_46: "f32[768]", primals_48: "f32[3072, 768]", primals_50: "f32[768, 3072]", primals_52: "f32[768]", primals_54: "f32[2304, 768]", primals_56: "f32[768, 768]", primals_58: "f32[768]", primals_60: "f32[3072, 768]", primals_62: "f32[768, 3072]", primals_64: "f32[768]", primals_66: "f32[2304, 768]", primals_68: "f32[768, 768]", primals_70: "f32[768]", primals_72: "f32[3072, 768]", primals_74: "f32[768, 3072]", primals_76: "f32[768]", primals_78: "f32[2304, 768]", primals_80: "f32[768, 768]", primals_82: "f32[768]", primals_84: "f32[3072, 768]", primals_86: "f32[768, 3072]", primals_88: "f32[768]", primals_90: "f32[2304, 768]", primals_92: "f32[768, 768]", primals_94: "f32[768]", primals_96: "f32[3072, 768]", primals_98: "f32[768, 3072]", primals_100: "f32[768]", primals_102: "f32[2304, 768]", primals_104: "f32[768, 768]", primals_106: "f32[768]", primals_108: "f32[3072, 768]", primals_110: "f32[768, 3072]", primals_112: "f32[768]", primals_114: "f32[2304, 768]", primals_116: "f32[768, 768]", primals_118: "f32[768]", primals_120: "f32[3072, 768]", primals_122: "f32[768, 3072]", primals_124: "f32[768]", primals_126: "f32[2304, 768]", primals_128: "f32[768, 768]", primals_130: "f32[768]", primals_132: "f32[3072, 768]", primals_134: "f32[768, 3072]", primals_136: "f32[768]", primals_138: "f32[2304, 768]", primals_140: "f32[768, 768]", primals_142: "f32[768]", primals_144: "f32[3072, 768]", primals_146: "f32[768, 3072]", primals_148: "f32[768]", unsqueeze: "i64[1, 64]", mul: "f32[1, 64, 768]", view: "f32[64, 768]", permute_1: "f32[1, 12, 64, 64]", permute_2: "f32[1, 12, 64, 64]", permute_3: "f32[1, 12, 64, 64]", getitem_5: "f32[1, 12, 64, 64]", getitem_6: "f32[1, 12, 64]", getitem_7: "i64[]", getitem_8: "i64[]", mul_2: "f32[1, 64, 768]", view_8: "f32[64, 768]", addmm_2: "f32[64, 3072]", view_10: "f32[64, 3072]", mul_8: "f32[1, 64, 768]", view_12: "f32[64, 768]", permute_9: "f32[1, 12, 64, 64]", permute_10: "f32[1, 12, 64, 64]", permute_11: "f32[1, 12, 64, 64]", getitem_16: "f32[1, 12, 64, 64]", getitem_17: "f32[1, 12, 64]", getitem_18: "i64[]", getitem_19: "i64[]", mul_10: "f32[1, 64, 768]", view_20: "f32[64, 768]", addmm_6: "f32[64, 3072]", view_22: "f32[64, 3072]", mul_16: "f32[1, 64, 768]", view_24: "f32[64, 768]", permute_17: "f32[1, 12, 64, 64]", permute_18: "f32[1, 12, 64, 64]", permute_19: "f32[1, 12, 64, 64]", getitem_27: "f32[1, 12, 64, 64]", getitem_28: "f32[1, 12, 64]", getitem_29: "i64[]", getitem_30: "i64[]", mul_18: "f32[1, 64, 768]", view_32: "f32[64, 768]", addmm_10: "f32[64, 3072]", view_34: "f32[64, 3072]", mul_24: "f32[1, 64, 768]", view_36: "f32[64, 768]", permute_25: "f32[1, 12, 64, 64]", permute_26: "f32[1, 12, 64, 64]", permute_27: "f32[1, 12, 64, 64]", getitem_38: "f32[1, 12, 64, 64]", getitem_39: "f32[1, 12, 64]", getitem_40: "i64[]", getitem_41: "i64[]", mul_26: "f32[1, 64, 768]", view_44: "f32[64, 768]", addmm_14: "f32[64, 3072]", view_46: "f32[64, 3072]", mul_32: "f32[1, 64, 768]", view_48: "f32[64, 768]", permute_33: "f32[1, 12, 64, 64]", permute_34: "f32[1, 12, 64, 64]", permute_35: "f32[1, 12, 64, 64]", getitem_49: "f32[1, 12, 64, 64]", getitem_50: "f32[1, 12, 64]", getitem_51: "i64[]", getitem_52: "i64[]", mul_34: "f32[1, 64, 768]", view_56: "f32[64, 768]", addmm_18: "f32[64, 3072]", view_58: "f32[64, 3072]", mul_40: "f32[1, 64, 768]", view_60: "f32[64, 768]", permute_41: "f32[1, 12, 64, 64]", permute_42: "f32[1, 12, 64, 64]", permute_43: "f32[1, 12, 64, 64]", getitem_60: "f32[1, 12, 64, 64]", getitem_61: "f32[1, 12, 64]", getitem_62: "i64[]", getitem_63: "i64[]", mul_42: "f32[1, 64, 768]", view_68: "f32[64, 768]", addmm_22: "f32[64, 3072]", view_70: "f32[64, 3072]", mul_48: "f32[1, 64, 768]", view_72: "f32[64, 768]", permute_49: "f32[1, 12, 64, 64]", permute_50: "f32[1, 12, 64, 64]", permute_51: "f32[1, 12, 64, 64]", getitem_71: "f32[1, 12, 64, 64]", getitem_72: "f32[1, 12, 64]", getitem_73: "i64[]", getitem_74: "i64[]", mul_50: "f32[1, 64, 768]", view_80: "f32[64, 768]", addmm_26: "f32[64, 3072]", view_82: "f32[64, 3072]", mul_56: "f32[1, 64, 768]", view_84: "f32[64, 768]", permute_57: "f32[1, 12, 64, 64]", permute_58: "f32[1, 12, 64, 64]", permute_59: "f32[1, 12, 64, 64]", getitem_82: "f32[1, 12, 64, 64]", getitem_83: "f32[1, 12, 64]", getitem_84: "i64[]", getitem_85: "i64[]", mul_58: "f32[1, 64, 768]", view_92: "f32[64, 768]", addmm_30: "f32[64, 3072]", view_94: "f32[64, 3072]", mul_64: "f32[1, 64, 768]", view_96: "f32[64, 768]", permute_65: "f32[1, 12, 64, 64]", permute_66: "f32[1, 12, 64, 64]", permute_67: "f32[1, 12, 64, 64]", getitem_93: "f32[1, 12, 64, 64]", getitem_94: "f32[1, 12, 64]", getitem_95: "i64[]", getitem_96: "i64[]", mul_66: "f32[1, 64, 768]", view_104: "f32[64, 768]", addmm_34: "f32[64, 3072]", view_106: "f32[64, 3072]", mul_72: "f32[1, 64, 768]", view_108: "f32[64, 768]", permute_73: "f32[1, 12, 64, 64]", permute_74: "f32[1, 12, 64, 64]", permute_75: "f32[1, 12, 64, 64]", getitem_104: "f32[1, 12, 64, 64]", getitem_105: "f32[1, 12, 64]", getitem_106: "i64[]", getitem_107: "i64[]", mul_74: "f32[1, 64, 768]", view_116: "f32[64, 768]", addmm_38: "f32[64, 3072]", view_118: "f32[64, 3072]", mul_80: "f32[1, 64, 768]", view_120: "f32[64, 768]", permute_81: "f32[1, 12, 64, 64]", permute_82: "f32[1, 12, 64, 64]", permute_83: "f32[1, 12, 64, 64]", getitem_115: "f32[1, 12, 64, 64]", getitem_116: "f32[1, 12, 64]", getitem_117: "i64[]", getitem_118: "i64[]", mul_82: "f32[1, 64, 768]", view_128: "f32[64, 768]", addmm_42: "f32[64, 3072]", view_130: "f32[64, 3072]", mul_88: "f32[1, 64, 768]", view_132: "f32[64, 768]", permute_89: "f32[1, 12, 64, 64]", permute_90: "f32[1, 12, 64, 64]", permute_91: "f32[1, 12, 64, 64]", getitem_126: "f32[1, 12, 64, 64]", getitem_127: "f32[1, 12, 64]", getitem_128: "i64[]", getitem_129: "i64[]", mul_90: "f32[1, 64, 768]", view_140: "f32[64, 768]", addmm_46: "f32[64, 3072]", view_142: "f32[64, 3072]", mul_96: "f32[1, 64, 768]", full_default: "i64[1]", view_144: "f32[1, 768]", div: "f32[1, 64, 1]", div_1: "f32[1, 64, 1]", div_2: "f32[1, 64, 1]", div_3: "f32[1, 64, 1]", div_4: "f32[1, 64, 1]", div_5: "f32[1, 64, 1]", div_6: "f32[1, 64, 1]", div_7: "f32[1, 64, 1]", div_8: "f32[1, 64, 1]", div_9: "f32[1, 64, 1]", div_10: "f32[1, 64, 1]", div_11: "f32[1, 64, 1]", div_12: "f32[1, 64, 1]", div_13: "f32[1, 64, 1]", div_14: "f32[1, 64, 1]", div_15: "f32[1, 64, 1]", div_16: "f32[1, 64, 1]", div_17: "f32[1, 64, 1]", div_18: "f32[1, 64, 1]", div_19: "f32[1, 64, 1]", div_20: "f32[1, 64, 1]", div_21: "f32[1, 64, 1]", div_22: "f32[1, 64, 1]", div_23: "f32[1, 64, 1]", div_24: "f32[1, 64, 1]", tangents_1: "f32[1, 1, 50304]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:239 in forward, code: logits = self.lm_head(
        view_146: "f32[1, 50304]" = torch.ops.aten.reshape.default(tangents_1, [1, 50304]);  tangents_1 = None
        permute_97: "f32[50304, 1]" = torch.ops.aten.permute.default(view_146, [1, 0])
        mul_98: "f32[50304, 768]" = torch.ops.aten.mul.Tensor(permute_97, view_144);  permute_97 = view_144 = None
        permute_96: "f32[768, 50304]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_99: "f32[50304, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_1: "f32[1, 768]" = torch.ops.aten.mm.default(view_146, permute_99);  view_146 = permute_99 = None
        view_147: "f32[1, 1, 768]" = torch.ops.aten.reshape.default(mm_1, [1, 1, 768]);  mm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:240 in forward, code: x[:, [-1], :]
        full_default_1: "f32[1, 64, 768]" = torch.ops.aten.full.default([1, 64, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[1, 64, 768]" = torch.ops.aten.index_put.default(full_default_1, [None, full_default], view_147, True);  full_default_1 = full_default = view_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_100: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(index_put, primals_148);  primals_148 = None
        mul_101: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_100, 768)
        sum_1: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_100, [2], True)
        mul_102: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_100, mul_96);  mul_100 = None
        sum_2: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True);  mul_102 = None
        mul_103: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_2);  sum_2 = None
        sub_26: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_101, sum_1);  mul_101 = sum_1 = None
        sub_27: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_103);  sub_26 = mul_103 = None
        mul_104: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div, sub_27);  div = sub_27 = None
        mul_105: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(index_put, mul_96);  mul_96 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_105, [0, 1]);  mul_105 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(index_put, [0, 1]);  index_put = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_148: "f32[64, 768]" = torch.ops.aten.reshape.default(mul_104, [64, 768])
        permute_95: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_101: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        mm_2: "f32[64, 3072]" = torch.ops.aten.mm.default(view_148, permute_101);  permute_101 = None
        permute_102: "f32[768, 64]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_102, view_142);  permute_102 = view_142 = None
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True);  view_148 = None
        view_149: "f32[768]" = torch.ops.aten.reshape.default(sum_5, [768]);  sum_5 = None
        view_150: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_2, [1, 64, 3072]);  mm_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_141: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 64, 3072]);  addmm_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_92: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_106: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_150, mul_92);  mul_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_12: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_141, 3.0)
        mul_93: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_141, mul_93);  mul_93 = None
        mul_94: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_95: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_107: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_150, add_95);  view_150 = add_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_108: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_28: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_108);  mul_108 = None
        mul_109: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_106, sub_28);  mul_106 = sub_28 = None
        mul_110: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_109, 0.7978845608028654);  mul_109 = None
        mul_111: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_110, 0.044715)
        pow_13: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_141, 2.0);  view_141 = None
        mul_112: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
        mul_113: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
        add_99: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_110, mul_113);  mul_110 = mul_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_114: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_107, 0.5);  mul_107 = None
        add_100: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_99, mul_114);  add_99 = mul_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_151: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_100, [64, 3072]);  add_100 = None
        permute_94: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        mm_4: "f32[64, 768]" = torch.ops.aten.mm.default(view_151, permute_105);  permute_105 = None
        permute_106: "f32[3072, 64]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_106, view_140);  permute_106 = view_140 = None
        sum_6: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        view_152: "f32[3072]" = torch.ops.aten.reshape.default(sum_6, [3072]);  sum_6 = None
        view_153: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_4, [1, 64, 768]);  mm_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_116: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_153, primals_142);  primals_142 = None
        mul_117: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_116, 768)
        sum_7: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_116, [2], True)
        mul_118: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_116, mul_90);  mul_116 = None
        sum_8: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_118, [2], True);  mul_118 = None
        mul_119: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_8);  sum_8 = None
        sub_30: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_117, sum_7);  mul_117 = sum_7 = None
        sub_31: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_119);  sub_30 = mul_119 = None
        mul_120: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_31);  div_1 = sub_31 = None
        mul_121: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_153, mul_90);  mul_90 = None
        sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_121, [0, 1]);  mul_121 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_153, [0, 1]);  view_153 = None
        add_101: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_104, mul_120);  mul_104 = mul_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_154: "f32[64, 768]" = torch.ops.aten.reshape.default(add_101, [64, 768])
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        mm_6: "f32[64, 768]" = torch.ops.aten.mm.default(view_154, permute_109);  permute_109 = None
        permute_110: "f32[768, 64]" = torch.ops.aten.permute.default(view_154, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_92: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])
        view_137: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_92, [1, 64, 768]);  permute_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_138: "f32[64, 768]" = torch.ops.aten.reshape.default(view_137, [64, 768]);  view_137 = None
        mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_110, view_138);  permute_110 = view_138 = None
        sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        view_155: "f32[768]" = torch.ops.aten.reshape.default(sum_11, [768]);  sum_11 = None
        view_156: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 64, 768]);  mm_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_157: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_156, [1, 64, 12, 64]);  view_156 = None
        permute_113: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_113, permute_90, permute_89, permute_91, None, getitem_126, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False], True);  permute_113 = permute_90 = permute_89 = permute_91 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_134: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_135: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_136: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_114: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3]);  getitem_136 = None
        view_158: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_114, [1, 64, 768]);  permute_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_115: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_159: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_115, [1, 64, 768]);  permute_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_116: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
        view_160: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_116, [1, 64, 768]);  permute_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_159, view_160, view_158], 2);  view_159 = view_160 = view_158 = None
        view_161: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat, [64, 2304]);  cat = None
        permute_88: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_117: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_8: "f32[64, 768]" = torch.ops.aten.mm.default(view_161, permute_117);  permute_117 = None
        permute_118: "f32[2304, 64]" = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_9: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_118, view_132);  permute_118 = view_132 = None
        sum_12: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True);  view_161 = None
        view_162: "f32[2304]" = torch.ops.aten.reshape.default(sum_12, [2304]);  sum_12 = None
        view_163: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 64, 768]);  mm_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_123: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_163, primals_136);  primals_136 = None
        mul_124: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_123, 768)
        sum_13: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
        mul_125: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_123, mul_88);  mul_123 = None
        sum_14: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
        mul_126: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_88, sum_14);  sum_14 = None
        sub_33: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_124, sum_13);  mul_124 = sum_13 = None
        sub_34: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_33, mul_126);  sub_33 = mul_126 = None
        mul_127: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_34);  div_2 = sub_34 = None
        mul_128: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_163, mul_88);  mul_88 = None
        sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_163, [0, 1]);  view_163 = None
        add_102: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_101, mul_127);  add_101 = mul_127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_164: "f32[64, 768]" = torch.ops.aten.reshape.default(add_102, [64, 768])
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_121: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_10: "f32[64, 3072]" = torch.ops.aten.mm.default(view_164, permute_121);  permute_121 = None
        permute_122: "f32[768, 64]" = torch.ops.aten.permute.default(view_164, [1, 0])
        mm_11: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_122, view_130);  permute_122 = view_130 = None
        sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True);  view_164 = None
        view_165: "f32[768]" = torch.ops.aten.reshape.default(sum_17, [768]);  sum_17 = None
        view_166: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_10, [1, 64, 3072]);  mm_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_129: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_42, [1, 64, 3072]);  addmm_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_84: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_129: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_166, mul_84);  mul_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_11: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_85: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_129, mul_85);  mul_85 = None
        mul_86: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_87: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_130: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_166, add_87);  view_166 = add_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_131: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_35: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_131);  mul_131 = None
        mul_132: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_129, sub_35);  mul_129 = sub_35 = None
        mul_133: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_132, 0.7978845608028654);  mul_132 = None
        mul_134: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_133, 0.044715)
        pow_14: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 2.0);  view_129 = None
        mul_135: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_136: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        add_103: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_133, mul_136);  mul_133 = mul_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_137: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_130, 0.5);  mul_130 = None
        add_104: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_103, mul_137);  add_103 = mul_137 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_167: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_104, [64, 3072]);  add_104 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_12: "f32[64, 768]" = torch.ops.aten.mm.default(view_167, permute_125);  permute_125 = None
        permute_126: "f32[3072, 64]" = torch.ops.aten.permute.default(view_167, [1, 0])
        mm_13: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_126, view_128);  permute_126 = view_128 = None
        sum_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        view_168: "f32[3072]" = torch.ops.aten.reshape.default(sum_18, [3072]);  sum_18 = None
        view_169: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_12, [1, 64, 768]);  mm_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_139: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_169, primals_130);  primals_130 = None
        mul_140: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_139, 768)
        sum_19: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True)
        mul_141: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_139, mul_82);  mul_139 = None
        sum_20: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True);  mul_141 = None
        mul_142: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_20);  sum_20 = None
        sub_37: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_140, sum_19);  mul_140 = sum_19 = None
        sub_38: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_37, mul_142);  sub_37 = mul_142 = None
        mul_143: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_38);  div_3 = sub_38 = None
        mul_144: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_169, mul_82);  mul_82 = None
        sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1]);  mul_144 = None
        sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_169, [0, 1]);  view_169 = None
        add_105: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_102, mul_143);  add_102 = mul_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_170: "f32[64, 768]" = torch.ops.aten.reshape.default(add_105, [64, 768])
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_14: "f32[64, 768]" = torch.ops.aten.mm.default(view_170, permute_129);  permute_129 = None
        permute_130: "f32[768, 64]" = torch.ops.aten.permute.default(view_170, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_84: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])
        view_125: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_84, [1, 64, 768]);  permute_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_126: "f32[64, 768]" = torch.ops.aten.reshape.default(view_125, [64, 768]);  view_125 = None
        mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_130, view_126);  permute_130 = view_126 = None
        sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        view_171: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
        view_172: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_14, [1, 64, 768]);  mm_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_173: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_172, [1, 64, 12, 64]);  view_172 = None
        permute_133: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_133, permute_82, permute_81, permute_83, None, getitem_115, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False], True);  permute_133 = permute_82 = permute_81 = permute_83 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None
        getitem_138: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_139: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_140: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_134: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None
        view_174: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_134, [1, 64, 768]);  permute_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_135: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
        view_175: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_135, [1, 64, 768]);  permute_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_136: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_176: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_136, [1, 64, 768]);  permute_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_1: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_175, view_176, view_174], 2);  view_175 = view_176 = view_174 = None
        view_177: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_1, [64, 2304]);  cat_1 = None
        permute_80: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_137: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        mm_16: "f32[64, 768]" = torch.ops.aten.mm.default(view_177, permute_137);  permute_137 = None
        permute_138: "f32[2304, 64]" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_17: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_138, view_120);  permute_138 = view_120 = None
        sum_24: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
        view_178: "f32[2304]" = torch.ops.aten.reshape.default(sum_24, [2304]);  sum_24 = None
        view_179: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_16, [1, 64, 768]);  mm_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_146: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_179, primals_124);  primals_124 = None
        mul_147: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_146, 768)
        sum_25: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_146, [2], True)
        mul_148: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_146, mul_80);  mul_146 = None
        sum_26: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True);  mul_148 = None
        mul_149: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_26);  sum_26 = None
        sub_40: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_147, sum_25);  mul_147 = sum_25 = None
        sub_41: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_40, mul_149);  sub_40 = mul_149 = None
        mul_150: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_41);  div_4 = sub_41 = None
        mul_151: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_179, mul_80);  mul_80 = None
        sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_151, [0, 1]);  mul_151 = None
        sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1]);  view_179 = None
        add_106: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_105, mul_150);  add_105 = mul_150 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_180: "f32[64, 768]" = torch.ops.aten.reshape.default(add_106, [64, 768])
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_141: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_18: "f32[64, 3072]" = torch.ops.aten.mm.default(view_180, permute_141);  permute_141 = None
        permute_142: "f32[768, 64]" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_19: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_142, view_118);  permute_142 = view_118 = None
        sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        view_181: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
        view_182: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_18, [1, 64, 3072]);  mm_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_117: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_38, [1, 64, 3072]);  addmm_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_76: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
        mul_152: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_182, mul_76);  mul_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_10: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_117, 3.0)
        mul_77: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_117, mul_77);  mul_77 = None
        mul_78: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_79: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_153: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_182, add_79);  view_182 = add_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_154: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_42: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_154);  mul_154 = None
        mul_155: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_152, sub_42);  mul_152 = sub_42 = None
        mul_156: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_155, 0.7978845608028654);  mul_155 = None
        mul_157: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_156, 0.044715)
        pow_15: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_117, 2.0);  view_117 = None
        mul_158: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_159: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        add_107: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_156, mul_159);  mul_156 = mul_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_160: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_153, 0.5);  mul_153 = None
        add_108: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_107, mul_160);  add_107 = mul_160 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_183: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_108, [64, 3072]);  add_108 = None
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_20: "f32[64, 768]" = torch.ops.aten.mm.default(view_183, permute_145);  permute_145 = None
        permute_146: "f32[3072, 64]" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_21: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_146, view_116);  permute_146 = view_116 = None
        sum_30: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        view_184: "f32[3072]" = torch.ops.aten.reshape.default(sum_30, [3072]);  sum_30 = None
        view_185: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_20, [1, 64, 768]);  mm_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_162: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_185, primals_118);  primals_118 = None
        mul_163: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_162, 768)
        sum_31: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_162, [2], True)
        mul_164: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_162, mul_74);  mul_162 = None
        sum_32: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True);  mul_164 = None
        mul_165: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_32);  sum_32 = None
        sub_44: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_163, sum_31);  mul_163 = sum_31 = None
        sub_45: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_165);  sub_44 = mul_165 = None
        mul_166: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_45);  div_5 = sub_45 = None
        mul_167: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_185, mul_74);  mul_74 = None
        sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 1]);  mul_167 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
        add_109: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_106, mul_166);  add_106 = mul_166 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_186: "f32[64, 768]" = torch.ops.aten.reshape.default(add_109, [64, 768])
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_22: "f32[64, 768]" = torch.ops.aten.mm.default(view_186, permute_149);  permute_149 = None
        permute_150: "f32[768, 64]" = torch.ops.aten.permute.default(view_186, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_76: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])
        view_113: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_76, [1, 64, 768]);  permute_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_114: "f32[64, 768]" = torch.ops.aten.reshape.default(view_113, [64, 768]);  view_113 = None
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_150, view_114);  permute_150 = view_114 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        view_188: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_22, [1, 64, 768]);  mm_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_189: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_188, [1, 64, 12, 64]);  view_188 = None
        permute_153: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_153, permute_74, permute_73, permute_75, None, getitem_104, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False], True);  permute_153 = permute_74 = permute_73 = permute_75 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = None
        getitem_142: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_143: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_144: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_154: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
        view_190: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_154, [1, 64, 768]);  permute_154 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_155: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        view_191: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_155, [1, 64, 768]);  permute_155 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_156: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_143, [0, 2, 1, 3]);  getitem_143 = None
        view_192: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_156, [1, 64, 768]);  permute_156 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_2: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_191, view_192, view_190], 2);  view_191 = view_192 = view_190 = None
        view_193: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_2, [64, 2304]);  cat_2 = None
        permute_72: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_157: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        mm_24: "f32[64, 768]" = torch.ops.aten.mm.default(view_193, permute_157);  permute_157 = None
        permute_158: "f32[2304, 64]" = torch.ops.aten.permute.default(view_193, [1, 0])
        mm_25: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_158, view_108);  permute_158 = view_108 = None
        sum_36: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_193, [0], True);  view_193 = None
        view_194: "f32[2304]" = torch.ops.aten.reshape.default(sum_36, [2304]);  sum_36 = None
        view_195: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_24, [1, 64, 768]);  mm_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_169: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_195, primals_112);  primals_112 = None
        mul_170: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_169, 768)
        sum_37: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True)
        mul_171: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_169, mul_72);  mul_169 = None
        sum_38: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
        mul_172: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_38);  sum_38 = None
        sub_47: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_170, sum_37);  mul_170 = sum_37 = None
        sub_48: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_172);  sub_47 = mul_172 = None
        mul_173: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_48);  div_6 = sub_48 = None
        mul_174: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_195, mul_72);  mul_72 = None
        sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1]);  mul_174 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 1]);  view_195 = None
        add_110: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_109, mul_173);  add_109 = mul_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_196: "f32[64, 768]" = torch.ops.aten.reshape.default(add_110, [64, 768])
        permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_26: "f32[64, 3072]" = torch.ops.aten.mm.default(view_196, permute_161);  permute_161 = None
        permute_162: "f32[768, 64]" = torch.ops.aten.permute.default(view_196, [1, 0])
        mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_162, view_106);  permute_162 = view_106 = None
        sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
        view_197: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        view_198: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_26, [1, 64, 3072]);  mm_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_105: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 64, 3072]);  addmm_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_68: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
        mul_175: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_198, mul_68);  mul_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_9: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_105, 3.0)
        mul_69: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_105, mul_69);  mul_69 = None
        mul_70: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_71: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_176: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_198, add_71);  view_198 = add_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_177: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_49: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_177);  mul_177 = None
        mul_178: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_175, sub_49);  mul_175 = sub_49 = None
        mul_179: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_178, 0.7978845608028654);  mul_178 = None
        mul_180: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_179, 0.044715)
        pow_16: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_105, 2.0);  view_105 = None
        mul_181: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_182: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        add_111: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_179, mul_182);  mul_179 = mul_182 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_183: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_176, 0.5);  mul_176 = None
        add_112: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_111, mul_183);  add_111 = mul_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_199: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_112, [64, 3072]);  add_112 = None
        permute_70: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_165: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_28: "f32[64, 768]" = torch.ops.aten.mm.default(view_199, permute_165);  permute_165 = None
        permute_166: "f32[3072, 64]" = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_166, view_104);  permute_166 = view_104 = None
        sum_42: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        view_200: "f32[3072]" = torch.ops.aten.reshape.default(sum_42, [3072]);  sum_42 = None
        view_201: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_28, [1, 64, 768]);  mm_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_185: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_201, primals_106);  primals_106 = None
        mul_186: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
        sum_43: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
        mul_187: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_66);  mul_185 = None
        sum_44: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
        mul_188: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_44);  sum_44 = None
        sub_51: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_43);  mul_186 = sum_43 = None
        sub_52: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_188);  sub_51 = mul_188 = None
        mul_189: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_52);  div_7 = sub_52 = None
        mul_190: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_201, mul_66);  mul_66 = None
        sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 1]);  view_201 = None
        add_113: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_110, mul_189);  add_110 = mul_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_202: "f32[64, 768]" = torch.ops.aten.reshape.default(add_113, [64, 768])
        permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_30: "f32[64, 768]" = torch.ops.aten.mm.default(view_202, permute_169);  permute_169 = None
        permute_170: "f32[768, 64]" = torch.ops.aten.permute.default(view_202, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_68: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])
        view_101: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_68, [1, 64, 768]);  permute_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_102: "f32[64, 768]" = torch.ops.aten.reshape.default(view_101, [64, 768]);  view_101 = None
        mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_170, view_102);  permute_170 = view_102 = None
        sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_202, [0], True);  view_202 = None
        view_203: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
        view_204: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_30, [1, 64, 768]);  mm_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_205: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_204, [1, 64, 12, 64]);  view_204 = None
        permute_173: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_173, permute_66, permute_65, permute_67, None, getitem_93, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False], True);  permute_173 = permute_66 = permute_65 = permute_67 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        getitem_146: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_147: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_148: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_174: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        view_206: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_174, [1, 64, 768]);  permute_174 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_175: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
        view_207: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_175, [1, 64, 768]);  permute_175 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_176: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None
        view_208: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_176, [1, 64, 768]);  permute_176 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_3: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_207, view_208, view_206], 2);  view_207 = view_208 = view_206 = None
        view_209: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_3, [64, 2304]);  cat_3 = None
        permute_64: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_177: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_32: "f32[64, 768]" = torch.ops.aten.mm.default(view_209, permute_177);  permute_177 = None
        permute_178: "f32[2304, 64]" = torch.ops.aten.permute.default(view_209, [1, 0])
        mm_33: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_178, view_96);  permute_178 = view_96 = None
        sum_48: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
        view_210: "f32[2304]" = torch.ops.aten.reshape.default(sum_48, [2304]);  sum_48 = None
        view_211: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_32, [1, 64, 768]);  mm_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_192: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_211, primals_100);  primals_100 = None
        mul_193: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_49: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
        mul_194: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_64);  mul_192 = None
        sum_50: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_50);  sum_50 = None
        sub_54: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_49);  mul_193 = sum_49 = None
        sub_55: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_195);  sub_54 = mul_195 = None
        mul_196: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_55);  div_8 = sub_55 = None
        mul_197: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_211, mul_64);  mul_64 = None
        sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None
        add_114: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_113, mul_196);  add_113 = mul_196 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_212: "f32[64, 768]" = torch.ops.aten.reshape.default(add_114, [64, 768])
        permute_63: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_34: "f32[64, 3072]" = torch.ops.aten.mm.default(view_212, permute_181);  permute_181 = None
        permute_182: "f32[768, 64]" = torch.ops.aten.permute.default(view_212, [1, 0])
        mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_182, view_94);  permute_182 = view_94 = None
        sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
        view_213: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
        view_214: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_34, [1, 64, 3072]);  mm_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_93: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_30, [1, 64, 3072]);  addmm_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_60: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_198: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_214, mul_60);  mul_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_8: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_93, 3.0)
        mul_61: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_93, mul_61);  mul_61 = None
        mul_62: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_63: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_199: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_214, add_63);  view_214 = add_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_200: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_56: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_200);  mul_200 = None
        mul_201: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_198, sub_56);  mul_198 = sub_56 = None
        mul_202: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_201, 0.7978845608028654);  mul_201 = None
        mul_203: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_202, 0.044715)
        pow_17: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_93, 2.0);  view_93 = None
        mul_204: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_205: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        add_115: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_202, mul_205);  mul_202 = mul_205 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_206: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_199, 0.5);  mul_199 = None
        add_116: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_115, mul_206);  add_115 = mul_206 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_215: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_116, [64, 3072]);  add_116 = None
        permute_62: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_185: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        mm_36: "f32[64, 768]" = torch.ops.aten.mm.default(view_215, permute_185);  permute_185 = None
        permute_186: "f32[3072, 64]" = torch.ops.aten.permute.default(view_215, [1, 0])
        mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_186, view_92);  permute_186 = view_92 = None
        sum_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        view_216: "f32[3072]" = torch.ops.aten.reshape.default(sum_54, [3072]);  sum_54 = None
        view_217: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_36, [1, 64, 768]);  mm_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_208: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_217, primals_94);  primals_94 = None
        mul_209: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_208, 768)
        sum_55: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
        mul_210: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_208, mul_58);  mul_208 = None
        sum_56: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
        mul_211: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_56);  sum_56 = None
        sub_58: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_209, sum_55);  mul_209 = sum_55 = None
        sub_59: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_211);  sub_58 = mul_211 = None
        mul_212: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_59);  div_9 = sub_59 = None
        mul_213: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_58);  mul_58 = None
        sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
        sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
        add_117: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_114, mul_212);  add_114 = mul_212 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_218: "f32[64, 768]" = torch.ops.aten.reshape.default(add_117, [64, 768])
        permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_189: "f32[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        mm_38: "f32[64, 768]" = torch.ops.aten.mm.default(view_218, permute_189);  permute_189 = None
        permute_190: "f32[768, 64]" = torch.ops.aten.permute.default(view_218, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_60: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])
        view_89: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_60, [1, 64, 768]);  permute_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_90: "f32[64, 768]" = torch.ops.aten.reshape.default(view_89, [64, 768]);  view_89 = None
        mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_190, view_90);  permute_190 = view_90 = None
        sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        view_219: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
        view_220: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_38, [1, 64, 768]);  mm_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_221: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_220, [1, 64, 12, 64]);  view_220 = None
        permute_193: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_193, permute_58, permute_57, permute_59, None, getitem_82, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False], True);  permute_193 = permute_58 = permute_57 = permute_59 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_150: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_151: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_152: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_194: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
        view_222: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_194, [1, 64, 768]);  permute_194 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_195: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_223: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_195, [1, 64, 768]);  permute_195 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_196: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
        view_224: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_196, [1, 64, 768]);  permute_196 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_4: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_223, view_224, view_222], 2);  view_223 = view_224 = view_222 = None
        view_225: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_4, [64, 2304]);  cat_4 = None
        permute_56: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_197: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_40: "f32[64, 768]" = torch.ops.aten.mm.default(view_225, permute_197);  permute_197 = None
        permute_198: "f32[2304, 64]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_41: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_198, view_84);  permute_198 = view_84 = None
        sum_60: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
        view_226: "f32[2304]" = torch.ops.aten.reshape.default(sum_60, [2304]);  sum_60 = None
        view_227: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_40, [1, 64, 768]);  mm_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_215: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_227, primals_88);  primals_88 = None
        mul_216: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_215, 768)
        sum_61: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
        mul_217: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_215, mul_56);  mul_215 = None
        sum_62: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
        mul_218: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_62);  sum_62 = None
        sub_61: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_216, sum_61);  mul_216 = sum_61 = None
        sub_62: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_61, mul_218);  sub_61 = mul_218 = None
        mul_219: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_62);  div_10 = sub_62 = None
        mul_220: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_227, mul_56);  mul_56 = None
        sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_227, [0, 1]);  view_227 = None
        add_118: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_117, mul_219);  add_117 = mul_219 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_228: "f32[64, 768]" = torch.ops.aten.reshape.default(add_118, [64, 768])
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_201: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_42: "f32[64, 3072]" = torch.ops.aten.mm.default(view_228, permute_201);  permute_201 = None
        permute_202: "f32[768, 64]" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_202, view_82);  permute_202 = view_82 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
        view_229: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        view_230: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_42, [1, 64, 3072]);  mm_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_81: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_26, [1, 64, 3072]);  addmm_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_52: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        mul_221: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_230, mul_52);  mul_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_7: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 3.0)
        mul_53: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_81, mul_53);  mul_53 = None
        mul_54: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_55: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_222: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_230, add_55);  view_230 = add_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_223: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_63: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_223);  mul_223 = None
        mul_224: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_221, sub_63);  mul_221 = sub_63 = None
        mul_225: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_224, 0.7978845608028654);  mul_224 = None
        mul_226: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_225, 0.044715)
        pow_18: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 2.0);  view_81 = None
        mul_227: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_228: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
        add_119: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_225, mul_228);  mul_225 = mul_228 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_229: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_222, 0.5);  mul_222 = None
        add_120: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_119, mul_229);  add_119 = mul_229 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_231: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_120, [64, 3072]);  add_120 = None
        permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_44: "f32[64, 768]" = torch.ops.aten.mm.default(view_231, permute_205);  permute_205 = None
        permute_206: "f32[3072, 64]" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_206, view_80);  permute_206 = view_80 = None
        sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        view_232: "f32[3072]" = torch.ops.aten.reshape.default(sum_66, [3072]);  sum_66 = None
        view_233: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_44, [1, 64, 768]);  mm_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_231: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_233, primals_82);  primals_82 = None
        mul_232: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_231, 768)
        sum_67: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True)
        mul_233: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_231, mul_50);  mul_231 = None
        sum_68: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True);  mul_233 = None
        mul_234: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_68);  sum_68 = None
        sub_65: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_232, sum_67);  mul_232 = sum_67 = None
        sub_66: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_234);  sub_65 = mul_234 = None
        mul_235: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_66);  div_11 = sub_66 = None
        mul_236: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_50);  mul_50 = None
        sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 1]);  mul_236 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
        add_121: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_118, mul_235);  add_118 = mul_235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_234: "f32[64, 768]" = torch.ops.aten.reshape.default(add_121, [64, 768])
        permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_209: "f32[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_46: "f32[64, 768]" = torch.ops.aten.mm.default(view_234, permute_209);  permute_209 = None
        permute_210: "f32[768, 64]" = torch.ops.aten.permute.default(view_234, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_52: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])
        view_77: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_52, [1, 64, 768]);  permute_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_78: "f32[64, 768]" = torch.ops.aten.reshape.default(view_77, [64, 768]);  view_77 = None
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_210, view_78);  permute_210 = view_78 = None
        sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        view_235: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
        view_236: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_46, [1, 64, 768]);  mm_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_237: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_236, [1, 64, 12, 64]);  view_236 = None
        permute_213: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_213, permute_50, permute_49, permute_51, None, getitem_71, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False], True);  permute_213 = permute_50 = permute_49 = permute_51 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = None
        getitem_154: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_155: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_156: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_214: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        view_238: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_214, [1, 64, 768]);  permute_214 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_215: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None
        view_239: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_215, [1, 64, 768]);  permute_215 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_216: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        view_240: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_216, [1, 64, 768]);  permute_216 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_5: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_239, view_240, view_238], 2);  view_239 = view_240 = view_238 = None
        view_241: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_5, [64, 2304]);  cat_5 = None
        permute_48: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_217: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_48: "f32[64, 768]" = torch.ops.aten.mm.default(view_241, permute_217);  permute_217 = None
        permute_218: "f32[2304, 64]" = torch.ops.aten.permute.default(view_241, [1, 0])
        mm_49: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_218, view_72);  permute_218 = view_72 = None
        sum_72: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
        view_242: "f32[2304]" = torch.ops.aten.reshape.default(sum_72, [2304]);  sum_72 = None
        view_243: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_48, [1, 64, 768]);  mm_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_238: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_243, primals_76);  primals_76 = None
        mul_239: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
        sum_73: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
        mul_240: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_48);  mul_238 = None
        sum_74: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
        mul_241: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_74);  sum_74 = None
        sub_68: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_73);  mul_239 = sum_73 = None
        sub_69: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_241);  sub_68 = mul_241 = None
        mul_242: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_69);  div_12 = sub_69 = None
        mul_243: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_243, mul_48);  mul_48 = None
        sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 1]);  view_243 = None
        add_122: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_121, mul_242);  add_121 = mul_242 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_244: "f32[64, 768]" = torch.ops.aten.reshape.default(add_122, [64, 768])
        permute_47: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_221: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_50: "f32[64, 3072]" = torch.ops.aten.mm.default(view_244, permute_221);  permute_221 = None
        permute_222: "f32[768, 64]" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_222, view_70);  permute_222 = view_70 = None
        sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
        view_245: "f32[768]" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
        view_246: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_50, [1, 64, 3072]);  mm_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_69: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 64, 3072]);  addmm_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_44: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
        mul_244: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_246, mul_44);  mul_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_6: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_69, 3.0)
        mul_45: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_69, mul_45);  mul_45 = None
        mul_46: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_47: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_245: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_246, add_47);  view_246 = add_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_246: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_70: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_246);  mul_246 = None
        mul_247: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_244, sub_70);  mul_244 = sub_70 = None
        mul_248: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_247, 0.7978845608028654);  mul_247 = None
        mul_249: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_248, 0.044715)
        pow_19: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_69, 2.0);  view_69 = None
        mul_250: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_251: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        add_123: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_248, mul_251);  mul_248 = mul_251 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_252: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_245, 0.5);  mul_245 = None
        add_124: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_123, mul_252);  add_123 = mul_252 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_247: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_124, [64, 3072]);  add_124 = None
        permute_46: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_225: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_52: "f32[64, 768]" = torch.ops.aten.mm.default(view_247, permute_225);  permute_225 = None
        permute_226: "f32[3072, 64]" = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_226, view_68);  permute_226 = view_68 = None
        sum_78: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True);  view_247 = None
        view_248: "f32[3072]" = torch.ops.aten.reshape.default(sum_78, [3072]);  sum_78 = None
        view_249: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_52, [1, 64, 768]);  mm_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_254: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_249, primals_70);  primals_70 = None
        mul_255: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_254, 768)
        sum_79: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
        mul_256: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_254, mul_42);  mul_254 = None
        sum_80: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
        mul_257: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_80);  sum_80 = None
        sub_72: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_255, sum_79);  mul_255 = sum_79 = None
        sub_73: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_257);  sub_72 = mul_257 = None
        mul_258: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_73);  div_13 = sub_73 = None
        mul_259: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_249, mul_42);  mul_42 = None
        sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
        sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 1]);  view_249 = None
        add_125: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_122, mul_258);  add_122 = mul_258 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_250: "f32[64, 768]" = torch.ops.aten.reshape.default(add_125, [64, 768])
        permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_229: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_54: "f32[64, 768]" = torch.ops.aten.mm.default(view_250, permute_229);  permute_229 = None
        permute_230: "f32[768, 64]" = torch.ops.aten.permute.default(view_250, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_44: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])
        view_65: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_44, [1, 64, 768]);  permute_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_66: "f32[64, 768]" = torch.ops.aten.reshape.default(view_65, [64, 768]);  view_65 = None
        mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_230, view_66);  permute_230 = view_66 = None
        sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
        view_251: "f32[768]" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        view_252: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_54, [1, 64, 768]);  mm_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_253: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_252, [1, 64, 12, 64]);  view_252 = None
        permute_233: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_233, permute_42, permute_41, permute_43, None, getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False], True);  permute_233 = permute_42 = permute_41 = permute_43 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_158: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_159: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_160: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_234: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
        view_254: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_234, [1, 64, 768]);  permute_234 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_235: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
        view_255: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_235, [1, 64, 768]);  permute_235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_236: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
        view_256: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_236, [1, 64, 768]);  permute_236 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_6: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_255, view_256, view_254], 2);  view_255 = view_256 = view_254 = None
        view_257: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_6, [64, 2304]);  cat_6 = None
        permute_40: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_237: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        mm_56: "f32[64, 768]" = torch.ops.aten.mm.default(view_257, permute_237);  permute_237 = None
        permute_238: "f32[2304, 64]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_57: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_238, view_60);  permute_238 = view_60 = None
        sum_84: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
        view_258: "f32[2304]" = torch.ops.aten.reshape.default(sum_84, [2304]);  sum_84 = None
        view_259: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_56, [1, 64, 768]);  mm_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_261: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_259, primals_64);  primals_64 = None
        mul_262: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_261, 768)
        sum_85: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
        mul_263: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_261, mul_40);  mul_261 = None
        sum_86: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
        mul_264: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_86);  sum_86 = None
        sub_75: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_262, sum_85);  mul_262 = sum_85 = None
        sub_76: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_264);  sub_75 = mul_264 = None
        mul_265: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_76);  div_14 = sub_76 = None
        mul_266: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_259, mul_40);  mul_40 = None
        sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
        sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_259, [0, 1]);  view_259 = None
        add_126: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_125, mul_265);  add_125 = mul_265 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_260: "f32[64, 768]" = torch.ops.aten.reshape.default(add_126, [64, 768])
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        mm_58: "f32[64, 3072]" = torch.ops.aten.mm.default(view_260, permute_241);  permute_241 = None
        permute_242: "f32[768, 64]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_59: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_58);  permute_242 = view_58 = None
        sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
        view_261: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
        view_262: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_58, [1, 64, 3072]);  mm_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_57: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_18, [1, 64, 3072]);  addmm_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_36: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_57, 0.5)
        mul_267: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_262, mul_36);  mul_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_5: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_57, 3.0)
        mul_37: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_57, mul_37);  mul_37 = None
        mul_38: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_39: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_268: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_262, add_39);  view_262 = add_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_269: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_77: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_269);  mul_269 = None
        mul_270: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_267, sub_77);  mul_267 = sub_77 = None
        mul_271: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_270, 0.7978845608028654);  mul_270 = None
        mul_272: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_271, 0.044715)
        pow_20: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_57, 2.0);  view_57 = None
        mul_273: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_274: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        add_127: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_271, mul_274);  mul_271 = mul_274 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_275: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_268, 0.5);  mul_268 = None
        add_128: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_127, mul_275);  add_127 = mul_275 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_263: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_128, [64, 3072]);  add_128 = None
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_60: "f32[64, 768]" = torch.ops.aten.mm.default(view_263, permute_245);  permute_245 = None
        permute_246: "f32[3072, 64]" = torch.ops.aten.permute.default(view_263, [1, 0])
        mm_61: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_56);  permute_246 = view_56 = None
        sum_90: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
        view_264: "f32[3072]" = torch.ops.aten.reshape.default(sum_90, [3072]);  sum_90 = None
        view_265: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_60, [1, 64, 768]);  mm_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_277: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_265, primals_58);  primals_58 = None
        mul_278: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_277, 768)
        sum_91: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
        mul_279: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_277, mul_34);  mul_277 = None
        sum_92: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        mul_280: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_92);  sum_92 = None
        sub_79: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_278, sum_91);  mul_278 = sum_91 = None
        sub_80: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_79, mul_280);  sub_79 = mul_280 = None
        mul_281: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_80);  div_15 = sub_80 = None
        mul_282: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_34);  mul_34 = None
        sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
        sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None
        add_129: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_126, mul_281);  add_126 = mul_281 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_266: "f32[64, 768]" = torch.ops.aten.reshape.default(add_129, [64, 768])
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_62: "f32[64, 768]" = torch.ops.aten.mm.default(view_266, permute_249);  permute_249 = None
        permute_250: "f32[768, 64]" = torch.ops.aten.permute.default(view_266, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_36: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        view_53: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_36, [1, 64, 768]);  permute_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_54: "f32[64, 768]" = torch.ops.aten.reshape.default(view_53, [64, 768]);  view_53 = None
        mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_54);  permute_250 = view_54 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
        view_267: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
        view_268: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_62, [1, 64, 768]);  mm_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_269: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_268, [1, 64, 12, 64]);  view_268 = None
        permute_253: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_253, permute_34, permute_33, permute_35, None, getitem_49, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False], True);  permute_253 = permute_34 = permute_33 = permute_35 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_162: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_163: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_164: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_254: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_164, [0, 2, 1, 3]);  getitem_164 = None
        view_270: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_254, [1, 64, 768]);  permute_254 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_255: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        view_271: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_255, [1, 64, 768]);  permute_255 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_256: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None
        view_272: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_256, [1, 64, 768]);  permute_256 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_7: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_271, view_272, view_270], 2);  view_271 = view_272 = view_270 = None
        view_273: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_7, [64, 2304]);  cat_7 = None
        permute_32: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        permute_257: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_64: "f32[64, 768]" = torch.ops.aten.mm.default(view_273, permute_257);  permute_257 = None
        permute_258: "f32[2304, 64]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_65: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_258, view_48);  permute_258 = view_48 = None
        sum_96: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        view_274: "f32[2304]" = torch.ops.aten.reshape.default(sum_96, [2304]);  sum_96 = None
        view_275: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_64, [1, 64, 768]);  mm_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_284: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_275, primals_52);  primals_52 = None
        mul_285: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_284, 768)
        sum_97: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True)
        mul_286: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_284, mul_32);  mul_284 = None
        sum_98: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True);  mul_286 = None
        mul_287: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_98);  sum_98 = None
        sub_82: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_285, sum_97);  mul_285 = sum_97 = None
        sub_83: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_82, mul_287);  sub_82 = mul_287 = None
        mul_288: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_83);  div_16 = sub_83 = None
        mul_289: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_275, mul_32);  mul_32 = None
        sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1]);  mul_289 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None
        add_130: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_129, mul_288);  add_129 = mul_288 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_276: "f32[64, 768]" = torch.ops.aten.reshape.default(add_130, [64, 768])
        permute_31: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_66: "f32[64, 3072]" = torch.ops.aten.mm.default(view_276, permute_261);  permute_261 = None
        permute_262: "f32[768, 64]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_67: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_262, view_46);  permute_262 = view_46 = None
        sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        view_277: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
        view_278: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_66, [1, 64, 3072]);  mm_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_45: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_14, [1, 64, 3072]);  addmm_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_28: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_290: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_278, mul_28);  mul_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_4: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_45, 3.0)
        mul_29: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_45, mul_29);  mul_29 = None
        mul_30: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_31: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_291: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_278, add_31);  view_278 = add_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_292: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_84: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_292);  mul_292 = None
        mul_293: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_290, sub_84);  mul_290 = sub_84 = None
        mul_294: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_293, 0.7978845608028654);  mul_293 = None
        mul_295: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_294, 0.044715)
        pow_21: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_45, 2.0);  view_45 = None
        mul_296: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_297: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        add_131: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_294, mul_297);  mul_294 = mul_297 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_298: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_291, 0.5);  mul_291 = None
        add_132: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_131, mul_298);  add_131 = mul_298 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_279: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_132, [64, 3072]);  add_132 = None
        permute_30: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_68: "f32[64, 768]" = torch.ops.aten.mm.default(view_279, permute_265);  permute_265 = None
        permute_266: "f32[3072, 64]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_69: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_266, view_44);  permute_266 = view_44 = None
        sum_102: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        view_280: "f32[3072]" = torch.ops.aten.reshape.default(sum_102, [3072]);  sum_102 = None
        view_281: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_68, [1, 64, 768]);  mm_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_300: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_281, primals_46);  primals_46 = None
        mul_301: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_300, 768)
        sum_103: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
        mul_302: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_26);  mul_300 = None
        sum_104: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
        mul_303: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_104);  sum_104 = None
        sub_86: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_301, sum_103);  mul_301 = sum_103 = None
        sub_87: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_303);  sub_86 = mul_303 = None
        mul_304: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_87);  div_17 = sub_87 = None
        mul_305: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_281, mul_26);  mul_26 = None
        sum_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_281, [0, 1]);  view_281 = None
        add_133: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_130, mul_304);  add_130 = mul_304 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_282: "f32[64, 768]" = torch.ops.aten.reshape.default(add_133, [64, 768])
        permute_29: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_70: "f32[64, 768]" = torch.ops.aten.mm.default(view_282, permute_269);  permute_269 = None
        permute_270: "f32[768, 64]" = torch.ops.aten.permute.default(view_282, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_28: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_41: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_28, [1, 64, 768]);  permute_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_42: "f32[64, 768]" = torch.ops.aten.reshape.default(view_41, [64, 768]);  view_41 = None
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_270, view_42);  permute_270 = view_42 = None
        sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283: "f32[768]" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
        view_284: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_70, [1, 64, 768]);  mm_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_285: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_284, [1, 64, 12, 64]);  view_284 = None
        permute_273: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_273, permute_26, permute_25, permute_27, None, getitem_38, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False], True);  permute_273 = permute_26 = permute_25 = permute_27 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_166: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_167: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_168: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_274: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None
        view_286: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_274, [1, 64, 768]);  permute_274 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_275: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
        view_287: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_275, [1, 64, 768]);  permute_275 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_276: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        view_288: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_276, [1, 64, 768]);  permute_276 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_8: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_287, view_288, view_286], 2);  view_287 = view_288 = view_286 = None
        view_289: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_8, [64, 2304]);  cat_8 = None
        permute_24: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_277: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_72: "f32[64, 768]" = torch.ops.aten.mm.default(view_289, permute_277);  permute_277 = None
        permute_278: "f32[2304, 64]" = torch.ops.aten.permute.default(view_289, [1, 0])
        mm_73: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_278, view_36);  permute_278 = view_36 = None
        sum_108: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
        view_290: "f32[2304]" = torch.ops.aten.reshape.default(sum_108, [2304]);  sum_108 = None
        view_291: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_72, [1, 64, 768]);  mm_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_307: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_291, primals_40);  primals_40 = None
        mul_308: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_307, 768)
        sum_109: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
        mul_309: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_307, mul_24);  mul_307 = None
        sum_110: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
        mul_310: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_110);  sum_110 = None
        sub_89: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_308, sum_109);  mul_308 = sum_109 = None
        sub_90: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_310);  sub_89 = mul_310 = None
        mul_311: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_90);  div_18 = sub_90 = None
        mul_312: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_291, mul_24);  mul_24 = None
        sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1]);  mul_312 = None
        sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_291, [0, 1]);  view_291 = None
        add_134: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_133, mul_311);  add_133 = mul_311 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_292: "f32[64, 768]" = torch.ops.aten.reshape.default(add_134, [64, 768])
        permute_23: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_281: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_74: "f32[64, 3072]" = torch.ops.aten.mm.default(view_292, permute_281);  permute_281 = None
        permute_282: "f32[768, 64]" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_282, view_34);  permute_282 = view_34 = None
        sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
        view_293: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
        view_294: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_74, [1, 64, 3072]);  mm_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_33: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 64, 3072]);  addmm_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_20: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
        mul_313: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_294, mul_20);  mul_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_3: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_33, 3.0)
        mul_21: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_33, mul_21);  mul_21 = None
        mul_22: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_23: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_314: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_294, add_23);  view_294 = add_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_315: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_91: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_315);  mul_315 = None
        mul_316: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_313, sub_91);  mul_313 = sub_91 = None
        mul_317: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_316, 0.7978845608028654);  mul_316 = None
        mul_318: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_317, 0.044715)
        pow_22: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_33, 2.0);  view_33 = None
        mul_319: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_320: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
        add_135: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_317, mul_320);  mul_317 = mul_320 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_321: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_314, 0.5);  mul_314 = None
        add_136: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_135, mul_321);  add_135 = mul_321 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_295: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_136, [64, 3072]);  add_136 = None
        permute_22: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_285: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_76: "f32[64, 768]" = torch.ops.aten.mm.default(view_295, permute_285);  permute_285 = None
        permute_286: "f32[3072, 64]" = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_286, view_32);  permute_286 = view_32 = None
        sum_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        view_296: "f32[3072]" = torch.ops.aten.reshape.default(sum_114, [3072]);  sum_114 = None
        view_297: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_76, [1, 64, 768]);  mm_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_323: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_297, primals_34);  primals_34 = None
        mul_324: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_323, 768)
        sum_115: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
        mul_325: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_323, mul_18);  mul_323 = None
        sum_116: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
        mul_326: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_116);  sum_116 = None
        sub_93: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_324, sum_115);  mul_324 = sum_115 = None
        sub_94: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_93, mul_326);  sub_93 = mul_326 = None
        mul_327: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_94);  div_19 = sub_94 = None
        mul_328: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_297, mul_18);  mul_18 = None
        sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
        sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 1]);  view_297 = None
        add_137: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_134, mul_327);  add_134 = mul_327 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_298: "f32[64, 768]" = torch.ops.aten.reshape.default(add_137, [64, 768])
        permute_21: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_78: "f32[64, 768]" = torch.ops.aten.mm.default(view_298, permute_289);  permute_289 = None
        permute_290: "f32[768, 64]" = torch.ops.aten.permute.default(view_298, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_20: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_29: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_20, [1, 64, 768]);  permute_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_30: "f32[64, 768]" = torch.ops.aten.reshape.default(view_29, [64, 768]);  view_29 = None
        mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_290, view_30);  permute_290 = view_30 = None
        sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        view_299: "f32[768]" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
        view_300: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_78, [1, 64, 768]);  mm_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_301: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_300, [1, 64, 12, 64]);  view_300 = None
        permute_293: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_293, permute_18, permute_17, permute_19, None, getitem_27, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False], True);  permute_293 = permute_18 = permute_17 = permute_19 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_170: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_171: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_172: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_294: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_302: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_294, [1, 64, 768]);  permute_294 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_295: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3]);  getitem_170 = None
        view_303: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_295, [1, 64, 768]);  permute_295 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_296: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None
        view_304: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_296, [1, 64, 768]);  permute_296 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_9: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_303, view_304, view_302], 2);  view_303 = view_304 = view_302 = None
        view_305: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_9, [64, 2304]);  cat_9 = None
        permute_16: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_297: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_80: "f32[64, 768]" = torch.ops.aten.mm.default(view_305, permute_297);  permute_297 = None
        permute_298: "f32[2304, 64]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_81: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_298, view_24);  permute_298 = view_24 = None
        sum_120: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        view_306: "f32[2304]" = torch.ops.aten.reshape.default(sum_120, [2304]);  sum_120 = None
        view_307: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_80, [1, 64, 768]);  mm_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_330: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_307, primals_28);  primals_28 = None
        mul_331: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_330, 768)
        sum_121: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True)
        mul_332: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_330, mul_16);  mul_330 = None
        sum_122: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        mul_333: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_122);  sum_122 = None
        sub_96: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_331, sum_121);  mul_331 = sum_121 = None
        sub_97: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_333);  sub_96 = mul_333 = None
        mul_334: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_97);  div_20 = sub_97 = None
        mul_335: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_307, mul_16);  mul_16 = None
        sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None
        add_138: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_137, mul_334);  add_137 = mul_334 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_308: "f32[64, 768]" = torch.ops.aten.reshape.default(add_138, [64, 768])
        permute_15: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_301: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_82: "f32[64, 3072]" = torch.ops.aten.mm.default(view_308, permute_301);  permute_301 = None
        permute_302: "f32[768, 64]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_83: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_302, view_22);  permute_302 = view_22 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
        view_309: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        view_310: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_82, [1, 64, 3072]);  mm_82 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_21: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_6, [1, 64, 3072]);  addmm_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_12: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_336: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_310, mul_12);  mul_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_2: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
        mul_13: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_21, mul_13);  mul_13 = None
        mul_14: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_15: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_337: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_310, add_15);  view_310 = add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_338: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_98: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_338);  mul_338 = None
        mul_339: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_336, sub_98);  mul_336 = sub_98 = None
        mul_340: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_339, 0.7978845608028654);  mul_339 = None
        mul_341: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_340, 0.044715)
        pow_23: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
        mul_342: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_343: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        add_139: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_340, mul_343);  mul_340 = mul_343 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_344: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_337, 0.5);  mul_337 = None
        add_140: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_139, mul_344);  add_139 = mul_344 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_311: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_140, [64, 3072]);  add_140 = None
        permute_14: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_305: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_84: "f32[64, 768]" = torch.ops.aten.mm.default(view_311, permute_305);  permute_305 = None
        permute_306: "f32[3072, 64]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_85: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_306, view_20);  permute_306 = view_20 = None
        sum_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312: "f32[3072]" = torch.ops.aten.reshape.default(sum_126, [3072]);  sum_126 = None
        view_313: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_84, [1, 64, 768]);  mm_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_346: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_313, primals_22);  primals_22 = None
        mul_347: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_346, 768)
        sum_127: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
        mul_348: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_346, mul_10);  mul_346 = None
        sum_128: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
        mul_349: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_128);  sum_128 = None
        sub_100: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_347, sum_127);  mul_347 = sum_127 = None
        sub_101: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_349);  sub_100 = mul_349 = None
        mul_350: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_101);  div_21 = sub_101 = None
        mul_351: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_10);  mul_10 = None
        sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
        add_141: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_138, mul_350);  add_138 = mul_350 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_314: "f32[64, 768]" = torch.ops.aten.reshape.default(add_141, [64, 768])
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_309: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_86: "f32[64, 768]" = torch.ops.aten.mm.default(view_314, permute_309);  permute_309 = None
        permute_310: "f32[768, 64]" = torch.ops.aten.permute.default(view_314, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_12: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])
        view_17: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_12, [1, 64, 768]);  permute_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_18: "f32[64, 768]" = torch.ops.aten.reshape.default(view_17, [64, 768]);  view_17 = None
        mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_310, view_18);  permute_310 = view_18 = None
        sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
        view_315: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
        view_316: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_86, [1, 64, 768]);  mm_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_317: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_316, [1, 64, 12, 64]);  view_316 = None
        permute_313: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_313, permute_10, permute_9, permute_11, None, getitem_16, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False], True);  permute_313 = permute_10 = permute_9 = permute_11 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_174: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_175: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_176: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_314: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        view_318: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_314, [1, 64, 768]);  permute_314 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_315: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
        view_319: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_315, [1, 64, 768]);  permute_315 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_316: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None
        view_320: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_316, [1, 64, 768]);  permute_316 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_10: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_319, view_320, view_318], 2);  view_319 = view_320 = view_318 = None
        view_321: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_10, [64, 2304]);  cat_10 = None
        permute_8: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_317: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_88: "f32[64, 768]" = torch.ops.aten.mm.default(view_321, permute_317);  permute_317 = None
        permute_318: "f32[2304, 64]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_89: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_318, view_12);  permute_318 = view_12 = None
        sum_132: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322: "f32[2304]" = torch.ops.aten.reshape.default(sum_132, [2304]);  sum_132 = None
        view_323: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_88, [1, 64, 768]);  mm_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_353: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_323, primals_16);  primals_16 = None
        mul_354: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_353, 768)
        sum_133: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
        mul_355: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_353, mul_8);  mul_353 = None
        sum_134: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
        mul_356: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_134);  sum_134 = None
        sub_103: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_354, sum_133);  mul_354 = sum_133 = None
        sub_104: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_103, mul_356);  sub_103 = mul_356 = None
        mul_357: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_104);  div_22 = sub_104 = None
        mul_358: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_323, mul_8);  mul_8 = None
        sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_323, [0, 1]);  view_323 = None
        add_142: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_141, mul_357);  add_141 = mul_357 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_324: "f32[64, 768]" = torch.ops.aten.reshape.default(add_142, [64, 768])
        permute_7: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_321: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_90: "f32[64, 3072]" = torch.ops.aten.mm.default(view_324, permute_321);  permute_321 = None
        permute_322: "f32[768, 64]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_91: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_322, view_10);  permute_322 = view_10 = None
        sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325: "f32[768]" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
        view_326: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(mm_90, [1, 64, 3072]);  mm_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_9: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_2, [1, 64, 3072]);  addmm_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_4: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_9, 0.5)
        mul_359: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_326, mul_4);  mul_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_1: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_9, 3.0)
        mul_5: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_9, mul_5);  mul_5 = None
        mul_6: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_7: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_360: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_326, add_7);  view_326 = add_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        mul_361: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_105: "f32[1, 64, 3072]" = torch.ops.aten.sub.Tensor(1, mul_361);  mul_361 = None
        mul_362: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_359, sub_105);  mul_359 = sub_105 = None
        mul_363: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_362, 0.7978845608028654);  mul_362 = None
        mul_364: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_363, 0.044715)
        pow_24: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_9, 2.0);  view_9 = None
        mul_365: "f32[1, 64, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_366: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
        add_143: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(mul_363, mul_366);  mul_363 = mul_366 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_367: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_360, 0.5);  mul_360 = None
        add_144: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(add_143, mul_367);  add_143 = mul_367 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_327: "f32[64, 3072]" = torch.ops.aten.reshape.default(add_144, [64, 3072]);  add_144 = None
        permute_6: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_325: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_92: "f32[64, 768]" = torch.ops.aten.mm.default(view_327, permute_325);  permute_325 = None
        permute_326: "f32[3072, 64]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_93: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_326, view_8);  permute_326 = view_8 = None
        sum_138: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        view_328: "f32[3072]" = torch.ops.aten.reshape.default(sum_138, [3072]);  sum_138 = None
        view_329: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_92, [1, 64, 768]);  mm_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_369: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_329, primals_10);  primals_10 = None
        mul_370: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_369, 768)
        sum_139: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
        mul_371: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_369, mul_2);  mul_369 = None
        sum_140: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
        mul_372: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_140);  sum_140 = None
        sub_107: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_370, sum_139);  mul_370 = sum_139 = None
        sub_108: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_372);  sub_107 = mul_372 = None
        mul_373: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_108);  div_23 = sub_108 = None
        mul_374: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_329, mul_2);  mul_2 = None
        sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
        sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_329, [0, 1]);  view_329 = None
        add_145: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_142, mul_373);  add_142 = mul_373 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_330: "f32[64, 768]" = torch.ops.aten.reshape.default(add_145, [64, 768])
        permute_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_94: "f32[64, 768]" = torch.ops.aten.mm.default(view_330, permute_329);  permute_329 = None
        permute_330: "f32[768, 64]" = torch.ops.aten.permute.default(view_330, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_4: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_5: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_4, [1, 64, 768]);  permute_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_6: "f32[64, 768]" = torch.ops.aten.reshape.default(view_5, [64, 768]);  view_5 = None
        mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_330, view_6);  permute_330 = view_6 = None
        sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        view_331: "f32[768]" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
        view_332: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_94, [1, 64, 768]);  mm_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        view_333: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(view_332, [1, 64, 12, 64]);  view_332 = None
        permute_333: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_333, permute_2, permute_1, permute_3, None, getitem_5, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False], True);  permute_333 = permute_2 = permute_1 = permute_3 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_178: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_179: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_180: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_334: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        view_334: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_334, [1, 64, 768]);  permute_334 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_335: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None
        view_335: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_335, [1, 64, 768]);  permute_335 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        permute_336: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
        view_336: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_336, [1, 64, 768]);  permute_336 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_11: "f32[1, 64, 2304]" = torch.ops.aten.cat.default([view_335, view_336, view_334], 2);  view_335 = view_336 = view_334 = None
        view_337: "f32[64, 2304]" = torch.ops.aten.reshape.default(cat_11, [64, 2304]);  cat_11 = None
        permute: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_337: "f32[2304, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_96: "f32[64, 768]" = torch.ops.aten.mm.default(view_337, permute_337);  permute_337 = None
        permute_338: "f32[2304, 64]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_97: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_338, view);  permute_338 = view = None
        sum_144: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        view_338: "f32[2304]" = torch.ops.aten.reshape.default(sum_144, [2304]);  sum_144 = None
        view_339: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_96, [1, 64, 768]);  mm_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_376: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_4);  primals_4 = None
        mul_377: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_376, 768)
        sum_145: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True)
        mul_378: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_376, mul);  mul_376 = None
        sum_146: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True);  mul_378 = None
        mul_379: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul, sum_146);  sum_146 = None
        sub_110: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_377, sum_145);  mul_377 = sum_145 = None
        sub_111: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_110, mul_379);  sub_110 = mul_379 = None
        mul_380: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_111);  div_24 = sub_111 = None
        mul_381: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_339, mul);  mul = None
        sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 1]);  mul_381 = None
        sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
        add_146: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_145, mul_380);  add_145 = mul_380 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:230 in forward, code: pos_emb = self.transformer.wpe(
        eq: "b8[1, 64]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_1: "b8[1, 64, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 64, 768]" = torch.ops.aten.where.self(unsqueeze_1, full_default_2, add_146);  unsqueeze_1 = None
        full_default_3: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_3, [unsqueeze], where, True);  full_default_3 = unsqueeze = where = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:229 in forward, code: tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        eq_1: "b8[1, 64]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_2: "b8[1, 64, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_1: "f32[1, 64, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_2, add_146);  unsqueeze_2 = full_default_2 = add_146 = None
        full_default_5: "f32[50304, 768]" = torch.ops.aten.full.default([50304, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_5, [primals_1], where_1, True);  full_default_5 = primals_1 = where_1 = None
        add_147: "f32[50304, 768]" = torch.ops.aten.add.Tensor(mul_98, index_put_2);  mul_98 = index_put_2 = None
        return (None, add_147, index_put_1, sum_147, sum_148, mm_97, view_338, mm_95, view_331, sum_141, sum_142, mm_93, view_328, mm_91, view_325, sum_135, sum_136, mm_89, view_322, mm_87, view_315, sum_129, sum_130, mm_85, view_312, mm_83, view_309, sum_123, sum_124, mm_81, view_306, mm_79, view_299, sum_117, sum_118, mm_77, view_296, mm_75, view_293, sum_111, sum_112, mm_73, view_290, mm_71, view_283, sum_105, sum_106, mm_69, view_280, mm_67, view_277, sum_99, sum_100, mm_65, view_274, mm_63, view_267, sum_93, sum_94, mm_61, view_264, mm_59, view_261, sum_87, sum_88, mm_57, view_258, mm_55, view_251, sum_81, sum_82, mm_53, view_248, mm_51, view_245, sum_75, sum_76, mm_49, view_242, mm_47, view_235, sum_69, sum_70, mm_45, view_232, mm_43, view_229, sum_63, sum_64, mm_41, view_226, mm_39, view_219, sum_57, sum_58, mm_37, view_216, mm_35, view_213, sum_51, sum_52, mm_33, view_210, mm_31, view_203, sum_45, sum_46, mm_29, view_200, mm_27, view_197, sum_39, sum_40, mm_25, view_194, mm_23, view_187, sum_33, sum_34, mm_21, view_184, mm_19, view_181, sum_27, sum_28, mm_17, view_178, mm_15, view_171, sum_21, sum_22, mm_13, view_168, mm_11, view_165, sum_15, sum_16, mm_9, view_162, mm_7, view_155, sum_9, sum_10, mm_5, view_152, mm_3, view_149, sum_3, sum_4)
