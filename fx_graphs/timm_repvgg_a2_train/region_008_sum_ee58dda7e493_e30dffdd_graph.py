class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", _shape_param_0, sum_3: "f32[1408]", squeeze_181: "f32[1408]", sum_5: "f32[1408]", squeeze_178: "f32[1408]", sum_7: "f32[384]", squeeze_175: "f32[384]", sum_9: "f32[384]", squeeze_172: "f32[384]", sum_11: "f32[384]", squeeze_169: "f32[384]", sum_13: "f32[384]", squeeze_166: "f32[384]", sum_15: "f32[384]", squeeze_163: "f32[384]", sum_17: "f32[384]", squeeze_160: "f32[384]", sum_19: "f32[384]", squeeze_157: "f32[384]", sum_21: "f32[384]", squeeze_154: "f32[384]", sum_23: "f32[384]", squeeze_151: "f32[384]", sum_25: "f32[384]", squeeze_148: "f32[384]", sum_27: "f32[384]", squeeze_145: "f32[384]", sum_29: "f32[384]", squeeze_142: "f32[384]", sum_31: "f32[384]", squeeze_139: "f32[384]", sum_33: "f32[384]", squeeze_136: "f32[384]", sum_35: "f32[384]", squeeze_133: "f32[384]", sum_37: "f32[384]", squeeze_130: "f32[384]", sum_39: "f32[384]", squeeze_127: "f32[384]", sum_41: "f32[384]", squeeze_124: "f32[384]", sum_43: "f32[384]", squeeze_121: "f32[384]", sum_45: "f32[384]", squeeze_118: "f32[384]", sum_47: "f32[384]", squeeze_115: "f32[384]", sum_49: "f32[384]", squeeze_112: "f32[384]", sum_51: "f32[384]", squeeze_109: "f32[384]", sum_53: "f32[384]", squeeze_106: "f32[384]", sum_55: "f32[384]", squeeze_103: "f32[384]", sum_57: "f32[384]", squeeze_100: "f32[384]", sum_59: "f32[384]", squeeze_97: "f32[384]", sum_61: "f32[384]", squeeze_94: "f32[384]", sum_63: "f32[384]", squeeze_91: "f32[384]", sum_65: "f32[384]", squeeze_88: "f32[384]", sum_67: "f32[384]", squeeze_85: "f32[384]", sum_69: "f32[384]", squeeze_82: "f32[384]", sum_71: "f32[384]", squeeze_79: "f32[384]", sum_73: "f32[384]", squeeze_76: "f32[384]", sum_75: "f32[384]", squeeze_73: "f32[384]", sum_77: "f32[384]", squeeze_70: "f32[384]", sum_79: "f32[384]", squeeze_67: "f32[384]", sum_81: "f32[384]", squeeze_64: "f32[384]", sum_83: "f32[384]", squeeze_61: "f32[384]", sum_85: "f32[384]", squeeze_58: "f32[384]", sum_87: "f32[384]", squeeze_55: "f32[384]", sum_89: "f32[192]", squeeze_52: "f32[192]", sum_91: "f32[192]", squeeze_49: "f32[192]", sum_93: "f32[192]", squeeze_46: "f32[192]", sum_95: "f32[192]", squeeze_43: "f32[192]", sum_97: "f32[192]", squeeze_40: "f32[192]", sum_99: "f32[192]", squeeze_37: "f32[192]", sum_101: "f32[192]", squeeze_34: "f32[192]", sum_103: "f32[192]", squeeze_31: "f32[192]", sum_105: "f32[192]", squeeze_28: "f32[192]", sum_107: "f32[192]", squeeze_25: "f32[192]", sum_109: "f32[192]", squeeze_22: "f32[192]", sum_111: "f32[96]", squeeze_19: "f32[96]", sum_113: "f32[96]", squeeze_16: "f32[96]", sum_115: "f32[96]", squeeze_13: "f32[96]", sum_117: "f32[96]", squeeze_10: "f32[96]", sum_119: "f32[96]", squeeze_7: "f32[96]", sum_121: "f32[64]", squeeze_4: "f32[64]", sum_123: "f32[64]", squeeze_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_181);  sum_3 = squeeze_181 = None
        mul_tensor_1: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_178);  sum_5 = squeeze_178 = None
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_175);  sum_7 = squeeze_175 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_172);  sum_9 = squeeze_172 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_169);  sum_11 = squeeze_169 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_166);  sum_13 = squeeze_166 = None
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_163);  sum_15 = squeeze_163 = None
        mul_tensor_7: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_160);  sum_17 = squeeze_160 = None
        mul_tensor_8: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_157);  sum_19 = squeeze_157 = None
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_154);  sum_21 = squeeze_154 = None
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_151);  sum_23 = squeeze_151 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_148);  sum_25 = squeeze_148 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_145);  sum_27 = squeeze_145 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_142);  sum_29 = squeeze_142 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_139);  sum_31 = squeeze_139 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_136);  sum_33 = squeeze_136 = None
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_133);  sum_35 = squeeze_133 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_130);  sum_37 = squeeze_130 = None
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_127);  sum_39 = squeeze_127 = None
        mul_tensor_19: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_124);  sum_41 = squeeze_124 = None
        mul_tensor_20: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_121);  sum_43 = squeeze_121 = None
        mul_tensor_21: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_118);  sum_45 = squeeze_118 = None
        mul_tensor_22: "f32[384]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_115);  sum_47 = squeeze_115 = None
        mul_tensor_23: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_112);  sum_49 = squeeze_112 = None
        mul_tensor_24: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_109);  sum_51 = squeeze_109 = None
        mul_tensor_25: "f32[384]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_106);  sum_53 = squeeze_106 = None
        mul_tensor_26: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_103);  sum_55 = squeeze_103 = None
        mul_tensor_27: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_100);  sum_57 = squeeze_100 = None
        mul_tensor_28: "f32[384]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_97);  sum_59 = squeeze_97 = None
        mul_tensor_29: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_94);  sum_61 = squeeze_94 = None
        mul_tensor_30: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_91);  sum_63 = squeeze_91 = None
        mul_tensor_31: "f32[384]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_88);  sum_65 = squeeze_88 = None
        mul_tensor_32: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_85);  sum_67 = squeeze_85 = None
        mul_tensor_33: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_82);  sum_69 = squeeze_82 = None
        mul_tensor_34: "f32[384]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_79);  sum_71 = squeeze_79 = None
        mul_tensor_35: "f32[384]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None
        mul_tensor_36: "f32[384]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None
        mul_tensor_37: "f32[384]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None
        mul_tensor_38: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_67);  sum_79 = squeeze_67 = None
        mul_tensor_39: "f32[384]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_64);  sum_81 = squeeze_64 = None
        mul_tensor_40: "f32[384]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_61);  sum_83 = squeeze_61 = None
        mul_tensor_41: "f32[384]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_58);  sum_85 = squeeze_58 = None
        mul_tensor_42: "f32[384]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_55);  sum_87 = squeeze_55 = None
        mul_tensor_43: "f32[192]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_52);  sum_89 = squeeze_52 = None
        mul_tensor_44: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_49);  sum_91 = squeeze_49 = None
        mul_tensor_45: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_46);  sum_93 = squeeze_46 = None
        mul_tensor_46: "f32[192]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_43);  sum_95 = squeeze_43 = None
        mul_tensor_47: "f32[192]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_40);  sum_97 = squeeze_40 = None
        mul_tensor_48: "f32[192]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_37);  sum_99 = squeeze_37 = None
        mul_tensor_49: "f32[192]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_34);  sum_101 = squeeze_34 = None
        mul_tensor_50: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_31);  sum_103 = squeeze_31 = None
        mul_tensor_51: "f32[192]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_28);  sum_105 = squeeze_28 = None
        mul_tensor_52: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_25);  sum_107 = squeeze_25 = None
        mul_tensor_53: "f32[192]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_22);  sum_109 = squeeze_22 = None
        mul_tensor_54: "f32[96]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_19);  sum_111 = squeeze_19 = None
        mul_tensor_55: "f32[96]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_16);  sum_113 = squeeze_16 = None
        mul_tensor_56: "f32[96]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_13);  sum_115 = squeeze_13 = None
        mul_tensor_57: "f32[96]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_10);  sum_117 = squeeze_10 = None
        mul_tensor_58: "f32[96]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_7);  sum_119 = squeeze_7 = None
        mul_tensor_59: "f32[64]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_4);  sum_121 = squeeze_4 = None
        mul_tensor_60: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_1);  sum_123 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52, mul_tensor_53, mul_tensor_54, mul_tensor_55, mul_tensor_56, mul_tensor_57, mul_tensor_58, mul_tensor_59, mul_tensor_60)
