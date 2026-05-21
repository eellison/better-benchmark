class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", _shape_param_0, sum_3: "f32[1280]", squeeze_145: "f32[1280]", sum_5: "f32[320]", squeeze_142: "f32[320]", mul_384: "f32[128, 1152, 1, 1]", mul_388: "f32[128, 48, 1, 1]", sum_10: "f32[1152]", squeeze_139: "f32[1152]", sum_12: "f32[1152]", squeeze_136: "f32[1152]", sum_14: "f32[192]", squeeze_133: "f32[192]", mul_427: "f32[128, 1152, 1, 1]", mul_431: "f32[128, 48, 1, 1]", sum_19: "f32[1152]", squeeze_130: "f32[1152]", sum_21: "f32[1152]", squeeze_127: "f32[1152]", sum_23: "f32[192]", squeeze_124: "f32[192]", mul_470: "f32[128, 1152, 1, 1]", mul_474: "f32[128, 48, 1, 1]", sum_28: "f32[1152]", squeeze_121: "f32[1152]", sum_30: "f32[1152]", squeeze_118: "f32[1152]", sum_32: "f32[192]", squeeze_115: "f32[192]", mul_513: "f32[128, 1152, 1, 1]", mul_517: "f32[128, 48, 1, 1]", sum_37: "f32[1152]", squeeze_112: "f32[1152]", sum_39: "f32[1152]", squeeze_109: "f32[1152]", sum_41: "f32[192]", squeeze_106: "f32[192]", mul_556: "f32[128, 672, 1, 1]", mul_560: "f32[128, 28, 1, 1]", sum_46: "f32[672]", squeeze_103: "f32[672]", sum_48: "f32[672]", squeeze_100: "f32[672]", sum_50: "f32[112]", squeeze_97: "f32[112]", mul_599: "f32[128, 672, 1, 1]", mul_603: "f32[128, 28, 1, 1]", sum_55: "f32[672]", squeeze_94: "f32[672]", sum_57: "f32[672]", squeeze_91: "f32[672]", sum_59: "f32[112]", squeeze_88: "f32[112]", mul_642: "f32[128, 672, 1, 1]", mul_646: "f32[128, 28, 1, 1]", sum_64: "f32[672]", squeeze_85: "f32[672]", sum_66: "f32[672]", squeeze_82: "f32[672]", sum_68: "f32[112]", squeeze_79: "f32[112]", mul_685: "f32[128, 480, 1, 1]", mul_689: "f32[128, 20, 1, 1]", sum_73: "f32[480]", squeeze_76: "f32[480]", sum_75: "f32[480]", squeeze_73: "f32[480]", sum_77: "f32[80]", squeeze_70: "f32[80]", mul_728: "f32[128, 480, 1, 1]", mul_732: "f32[128, 20, 1, 1]", sum_82: "f32[480]", squeeze_67: "f32[480]", sum_84: "f32[480]", squeeze_64: "f32[480]", sum_86: "f32[80]", squeeze_61: "f32[80]", mul_771: "f32[128, 480, 1, 1]", mul_775: "f32[128, 20, 1, 1]", sum_91: "f32[480]", squeeze_58: "f32[480]", sum_93: "f32[480]", squeeze_55: "f32[480]", sum_95: "f32[80]", squeeze_52: "f32[80]", mul_814: "f32[128, 240, 1, 1]", mul_818: "f32[128, 10, 1, 1]", sum_100: "f32[240]", squeeze_49: "f32[240]", sum_102: "f32[240]", squeeze_46: "f32[240]", sum_104: "f32[40]", squeeze_43: "f32[40]", mul_857: "f32[128, 240, 1, 1]", mul_861: "f32[128, 10, 1, 1]", sum_109: "f32[240]", squeeze_40: "f32[240]", sum_111: "f32[240]", squeeze_37: "f32[240]", sum_113: "f32[40]", squeeze_34: "f32[40]", mul_900: "f32[128, 144, 1, 1]", mul_904: "f32[128, 6, 1, 1]", sum_118: "f32[144]", squeeze_31: "f32[144]", sum_120: "f32[144]", squeeze_28: "f32[144]", sum_122: "f32[24]", squeeze_25: "f32[24]", mul_943: "f32[128, 144, 1, 1]", mul_947: "f32[128, 6, 1, 1]", sum_127: "f32[144]", squeeze_22: "f32[144]", sum_129: "f32[144]", squeeze_19: "f32[144]", sum_131: "f32[24]", squeeze_16: "f32[24]", mul_986: "f32[128, 96, 1, 1]", mul_990: "f32[128, 4, 1, 1]", sum_136: "f32[96]", squeeze_13: "f32[96]", sum_138: "f32[96]", squeeze_10: "f32[96]", sum_140: "f32[16]", squeeze_7: "f32[16]", mul_1029: "f32[128, 32, 1, 1]", mul_1033: "f32[128, 8, 1, 1]", sum_145: "f32[32]", squeeze_4: "f32[32]", sum_147: "f32[32]", squeeze_1: "f32[32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:344 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_145);  sum_3 = squeeze_145 = None
        mul_tensor_1: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_142);  sum_5 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_1: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3]);  mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_2: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2, 3]);  mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_2: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_139);  sum_10 = squeeze_139 = None
        mul_tensor_3: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_136);  sum_12 = squeeze_136 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_133);  sum_14 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_3: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_4: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_5: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
        mul_tensor_6: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_5: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_6: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 2, 3]);  mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_8: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_121);  sum_28 = squeeze_121 = None
        mul_tensor_9: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_118);  sum_30 = squeeze_118 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_115);  sum_32 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3]);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_8: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3]);  mul_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_11: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_112);  sum_37 = squeeze_112 = None
        mul_tensor_12: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_109);  sum_39 = squeeze_109 = None
        mul_tensor_13: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_106);  sum_41 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_9: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3]);  mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_10: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3]);  mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_14: "f32[672]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_103);  sum_46 = squeeze_103 = None
        mul_tensor_15: "f32[672]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_100);  sum_48 = squeeze_100 = None
        mul_tensor_16: "f32[112]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_97);  sum_50 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_11: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_12: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 2, 3]);  mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_17: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_94);  sum_55 = squeeze_94 = None
        mul_tensor_18: "f32[672]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_91);  sum_57 = squeeze_91 = None
        mul_tensor_19: "f32[112]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_88);  sum_59 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_13: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3]);  mul_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_14: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_646, [0, 2, 3]);  mul_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_20: "f32[672]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_85);  sum_64 = squeeze_85 = None
        mul_tensor_21: "f32[672]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_82);  sum_66 = squeeze_82 = None
        mul_tensor_22: "f32[112]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_79);  sum_68 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_15: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_16: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_23: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None
        mul_tensor_24: "f32[480]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None
        mul_tensor_25: "f32[80]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_17: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3]);  mul_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_18: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3]);  mul_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_26: "f32[480]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_67);  sum_82 = squeeze_67 = None
        mul_tensor_27: "f32[480]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_64);  sum_84 = squeeze_64 = None
        mul_tensor_28: "f32[80]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_61);  sum_86 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_19: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_20: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3]);  mul_775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_29: "f32[480]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_58);  sum_91 = squeeze_58 = None
        mul_tensor_30: "f32[480]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_55);  sum_93 = squeeze_55 = None
        mul_tensor_31: "f32[80]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_52);  sum_95 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_21: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_22: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 2, 3]);  mul_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_32: "f32[240]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_49);  sum_100 = squeeze_49 = None
        mul_tensor_33: "f32[240]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_46);  sum_102 = squeeze_46 = None
        mul_tensor_34: "f32[40]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_43);  sum_104 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_23: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3]);  mul_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_24: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2, 3]);  mul_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_35: "f32[240]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_40);  sum_109 = squeeze_40 = None
        mul_tensor_36: "f32[240]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_37);  sum_111 = squeeze_37 = None
        mul_tensor_37: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_34);  sum_113 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_25: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 2, 3]);  mul_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_26: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_38: "f32[144]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_31);  sum_118 = squeeze_31 = None
        mul_tensor_39: "f32[144]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_28);  sum_120 = squeeze_28 = None
        mul_tensor_40: "f32[24]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_25);  sum_122 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_27: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_943, [0, 2, 3]);  mul_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_28: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 2, 3]);  mul_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_41: "f32[144]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_22);  sum_127 = squeeze_22 = None
        mul_tensor_42: "f32[144]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_19);  sum_129 = squeeze_19 = None
        mul_tensor_43: "f32[24]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_16);  sum_131 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_29: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 2, 3]);  mul_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_30: "f32[4]" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 2, 3]);  mul_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_44: "f32[96]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_13);  sum_136 = squeeze_13 = None
        mul_tensor_45: "f32[96]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_10);  sum_138 = squeeze_10 = None
        mul_tensor_46: "f32[16]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_7);  sum_140 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_31: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3]);  mul_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_32: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1033, [0, 2, 3]);  mul_1033 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_47: "f32[32]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_4);  sum_145 = squeeze_4 = None
        mul_tensor_48: "f32[32]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_1);  sum_147 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, sum_dim_int_list_1, sum_dim_int_list_2, mul_tensor_2, mul_tensor_3, mul_tensor_4, sum_dim_int_list_3, sum_dim_int_list_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, sum_dim_int_list_5, sum_dim_int_list_6, mul_tensor_8, mul_tensor_9, mul_tensor_10, sum_dim_int_list_7, sum_dim_int_list_8, mul_tensor_11, mul_tensor_12, mul_tensor_13, sum_dim_int_list_9, sum_dim_int_list_10, mul_tensor_14, mul_tensor_15, mul_tensor_16, sum_dim_int_list_11, sum_dim_int_list_12, mul_tensor_17, mul_tensor_18, mul_tensor_19, sum_dim_int_list_13, sum_dim_int_list_14, mul_tensor_20, mul_tensor_21, mul_tensor_22, sum_dim_int_list_15, sum_dim_int_list_16, mul_tensor_23, mul_tensor_24, mul_tensor_25, sum_dim_int_list_17, sum_dim_int_list_18, mul_tensor_26, mul_tensor_27, mul_tensor_28, sum_dim_int_list_19, sum_dim_int_list_20, mul_tensor_29, mul_tensor_30, mul_tensor_31, sum_dim_int_list_21, sum_dim_int_list_22, mul_tensor_32, mul_tensor_33, mul_tensor_34, sum_dim_int_list_23, sum_dim_int_list_24, mul_tensor_35, mul_tensor_36, mul_tensor_37, sum_dim_int_list_25, sum_dim_int_list_26, mul_tensor_38, mul_tensor_39, mul_tensor_40, sum_dim_int_list_27, sum_dim_int_list_28, mul_tensor_41, mul_tensor_42, mul_tensor_43, sum_dim_int_list_29, sum_dim_int_list_30, mul_tensor_44, mul_tensor_45, mul_tensor_46, sum_dim_int_list_31, sum_dim_int_list_32, mul_tensor_47, mul_tensor_48)
