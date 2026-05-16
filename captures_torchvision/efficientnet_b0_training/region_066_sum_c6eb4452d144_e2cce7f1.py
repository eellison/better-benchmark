"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: c6eb4452d144
Shape hash: e2cce7f1
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 1000]", _shape_param_0, sum_3: "f32[1280]", squeeze_145: "f32[1280]", sum_5: "f32[320]", squeeze_142: "f32[320]", mul_395: "f32[4, 1152, 1, 1]", mul_399: "f32[4, 48, 1, 1]", sum_10: "f32[1152]", squeeze_139: "f32[1152]", sum_12: "f32[1152]", squeeze_136: "f32[1152]", sum_14: "f32[192]", squeeze_133: "f32[192]", mul_439: "f32[4, 1152, 1, 1]", mul_443: "f32[4, 48, 1, 1]", sum_19: "f32[1152]", squeeze_130: "f32[1152]", sum_21: "f32[1152]", squeeze_127: "f32[1152]", sum_23: "f32[192]", squeeze_124: "f32[192]", mul_483: "f32[4, 1152, 1, 1]", mul_487: "f32[4, 48, 1, 1]", sum_28: "f32[1152]", squeeze_121: "f32[1152]", sum_30: "f32[1152]", squeeze_118: "f32[1152]", sum_32: "f32[192]", squeeze_115: "f32[192]", mul_527: "f32[4, 1152, 1, 1]", mul_531: "f32[4, 48, 1, 1]", sum_37: "f32[1152]", squeeze_112: "f32[1152]", sum_39: "f32[1152]", squeeze_109: "f32[1152]", sum_41: "f32[192]", squeeze_106: "f32[192]", mul_570: "f32[4, 672, 1, 1]", mul_574: "f32[4, 28, 1, 1]", sum_46: "f32[672]", squeeze_103: "f32[672]", sum_48: "f32[672]", squeeze_100: "f32[672]", sum_50: "f32[112]", squeeze_97: "f32[112]", mul_614: "f32[4, 672, 1, 1]", mul_618: "f32[4, 28, 1, 1]", sum_55: "f32[672]", squeeze_94: "f32[672]", sum_57: "f32[672]", squeeze_91: "f32[672]", sum_59: "f32[112]", squeeze_88: "f32[112]", mul_658: "f32[4, 672, 1, 1]", mul_662: "f32[4, 28, 1, 1]", sum_64: "f32[672]", squeeze_85: "f32[672]", sum_66: "f32[672]", squeeze_82: "f32[672]", sum_68: "f32[112]", squeeze_79: "f32[112]", mul_701: "f32[4, 480, 1, 1]", mul_705: "f32[4, 20, 1, 1]", sum_73: "f32[480]", squeeze_76: "f32[480]", sum_75: "f32[480]", squeeze_73: "f32[480]", sum_77: "f32[80]", squeeze_70: "f32[80]", mul_745: "f32[4, 480, 1, 1]", mul_749: "f32[4, 20, 1, 1]", sum_82: "f32[480]", squeeze_67: "f32[480]", sum_84: "f32[480]", squeeze_64: "f32[480]", sum_86: "f32[80]", squeeze_61: "f32[80]", mul_789: "f32[4, 480, 1, 1]", mul_793: "f32[4, 20, 1, 1]", sum_91: "f32[480]", squeeze_58: "f32[480]", sum_93: "f32[480]", squeeze_55: "f32[480]", sum_95: "f32[80]", squeeze_52: "f32[80]", mul_832: "f32[4, 240, 1, 1]", mul_836: "f32[4, 10, 1, 1]", sum_100: "f32[240]", squeeze_49: "f32[240]", sum_102: "f32[240]", squeeze_46: "f32[240]", sum_104: "f32[40]", squeeze_43: "f32[40]", mul_876: "f32[4, 240, 1, 1]", mul_880: "f32[4, 10, 1, 1]", sum_109: "f32[240]", squeeze_40: "f32[240]", sum_111: "f32[240]", squeeze_37: "f32[240]", sum_113: "f32[40]", squeeze_34: "f32[40]", mul_919: "f32[4, 144, 1, 1]", mul_923: "f32[4, 6, 1, 1]", sum_118: "f32[144]", squeeze_31: "f32[144]", sum_120: "f32[144]", squeeze_28: "f32[144]", sum_122: "f32[24]", squeeze_25: "f32[24]", mul_963: "f32[4, 144, 1, 1]", mul_967: "f32[4, 6, 1, 1]", sum_127: "f32[144]", squeeze_22: "f32[144]", sum_129: "f32[144]", squeeze_19: "f32[144]", sum_131: "f32[24]", squeeze_16: "f32[24]", mul_1006: "f32[4, 96, 1, 1]", mul_1010: "f32[4, 4, 1, 1]", sum_136: "f32[96]", squeeze_13: "f32[96]", sum_138: "f32[96]", squeeze_10: "f32[96]", sum_140: "f32[16]", squeeze_7: "f32[16]", mul_1049: "f32[4, 32, 1, 1]", mul_1053: "f32[4, 8, 1, 1]", sum_145: "f32[32]", squeeze_4: "f32[32]", convolution: "f32[4, 32, 112, 112]", getitem_1: "f32[1, 32, 1, 1]", rsqrt: "f32[1, 32, 1, 1]", primals_6: "f32[32]", primals_7: "f32[32]", getitem_335: "f32[4, 32, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:339 in _forward_impl, code: x = self.classifier(x)
        permute_default: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:334 in _forward_impl, code: x = self.features(x)
        mul_tensor: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_145);  sum_3 = squeeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_1: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_142);  sum_5 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_1: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2, 3]);  mul_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_2: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3]);  mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_2: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_139);  sum_10 = squeeze_139 = None
        mul_tensor_3: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_136);  sum_12 = squeeze_136 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_133);  sum_14 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_3: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_4: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_443, [0, 2, 3]);  mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_5: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
        mul_tensor_6: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_5: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 2, 3]);  mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_6: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 2, 3]);  mul_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_8: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_121);  sum_28 = squeeze_121 = None
        mul_tensor_9: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_118);  sum_30 = squeeze_118 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_115);  sum_32 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_8: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 2, 3]);  mul_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_11: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_112);  sum_37 = squeeze_112 = None
        mul_tensor_12: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_109);  sum_39 = squeeze_109 = None
        mul_tensor_13: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_106);  sum_41 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_9: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 2, 3]);  mul_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_10: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3]);  mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_14: "f32[672]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_103);  sum_46 = squeeze_103 = None
        mul_tensor_15: "f32[672]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_100);  sum_48 = squeeze_100 = None
        mul_tensor_16: "f32[112]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_97);  sum_50 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_11: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_614, [0, 2, 3]);  mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_12: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3]);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_17: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_94);  sum_55 = squeeze_94 = None
        mul_tensor_18: "f32[672]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_91);  sum_57 = squeeze_91 = None
        mul_tensor_19: "f32[112]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_88);  sum_59 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_13: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3]);  mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_14: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_20: "f32[672]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_85);  sum_64 = squeeze_85 = None
        mul_tensor_21: "f32[672]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_82);  sum_66 = squeeze_82 = None
        mul_tensor_22: "f32[112]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_79);  sum_68 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_15: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_701, [0, 2, 3]);  mul_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_16: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_23: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None
        mul_tensor_24: "f32[480]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None
        mul_tensor_25: "f32[80]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_17: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 2, 3]);  mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_18: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3]);  mul_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_26: "f32[480]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_67);  sum_82 = squeeze_67 = None
        mul_tensor_27: "f32[480]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_64);  sum_84 = squeeze_64 = None
        mul_tensor_28: "f32[80]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_61);  sum_86 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_19: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3]);  mul_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_20: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3]);  mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_29: "f32[480]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_58);  sum_91 = squeeze_58 = None
        mul_tensor_30: "f32[480]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_55);  sum_93 = squeeze_55 = None
        mul_tensor_31: "f32[80]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_52);  sum_95 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_21: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 2, 3]);  mul_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_22: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 2, 3]);  mul_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_32: "f32[240]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_49);  sum_100 = squeeze_49 = None
        mul_tensor_33: "f32[240]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_46);  sum_102 = squeeze_46 = None
        mul_tensor_34: "f32[40]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_43);  sum_104 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_23: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 2, 3]);  mul_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_24: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2, 3]);  mul_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_35: "f32[240]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_40);  sum_109 = squeeze_40 = None
        mul_tensor_36: "f32[240]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_37);  sum_111 = squeeze_37 = None
        mul_tensor_37: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_34);  sum_113 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_25: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_26: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 2, 3]);  mul_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_38: "f32[144]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_31);  sum_118 = squeeze_31 = None
        mul_tensor_39: "f32[144]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_28);  sum_120 = squeeze_28 = None
        mul_tensor_40: "f32[24]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_25);  sum_122 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_27: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_963, [0, 2, 3]);  mul_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_28: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_41: "f32[144]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_22);  sum_127 = squeeze_22 = None
        mul_tensor_42: "f32[144]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_19);  sum_129 = squeeze_19 = None
        mul_tensor_43: "f32[24]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_16);  sum_131 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_29: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1006, [0, 2, 3]);  mul_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_30: "f32[4]" = torch.ops.aten.sum.dim_IntList(mul_1010, [0, 2, 3]);  mul_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_44: "f32[96]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_13);  sum_136 = squeeze_13 = None
        mul_tensor_45: "f32[96]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_10);  sum_138 = squeeze_10 = None
        mul_tensor_46: "f32[16]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_7);  sum_140 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_31: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1049, [0, 2, 3]);  mul_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_32: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1053, [0, 2, 3]);  mul_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        mul_tensor_47: "f32[32]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_4);  sum_145 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:334 in _forward_impl, code: x = self.features(x)
        sub_tensor: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_tensor_48: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_default_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_49: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_48, unsqueeze_default_1);  mul_tensor_48 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_default_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_49, unsqueeze_default_3);  mul_tensor_49 = unsqueeze_default_3 = None
        neg_default: "f32[4, 32, 112, 112]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[4, 32, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[4, 32, 112, 112]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_50: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_51: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_335, mul_tensor_50);  getitem_335 = None
        sub_tensor_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(1, mul_tensor_50);  mul_tensor_50 = None
        mul_tensor_52: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor_1);  add_tensor = sub_tensor_1 = None
        add_tensor_2: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_52, 1);  mul_tensor_52 = None
        mul_tensor_53: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_51, add_tensor_2);  mul_tensor_51 = add_tensor_2 = None
        squeeze_dims: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_default_4: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list_33: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 2, 3])
        sub_tensor_2: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_default_6);  convolution = unsqueeze_default_6 = None
        mul_tensor_54: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_53, sub_tensor_2)
        sum_dim_int_list_34: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 2, 3]);  mul_tensor_54 = None
        mul_tensor_55: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_33, 1.992984693877551e-05);  sum_dim_int_list_33 = None
        unsqueeze_default_7: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_55, 0);  mul_tensor_55 = None
        unsqueeze_default_8: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_56: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_34, 1.992984693877551e-05)
        squeeze_dims_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor_57: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_58: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_56, mul_tensor_57);  mul_tensor_56 = mul_tensor_57 = None
        unsqueeze_default_10: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_58, 0);  mul_tensor_58 = None
        unsqueeze_default_11: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_59: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_6);  primals_6 = None
        unsqueeze_default_13: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_59, 0);  mul_tensor_59 = None
        unsqueeze_default_14: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_60: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_12);  sub_tensor_2 = unsqueeze_default_12 = None
        sub_tensor_3: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_tensor_53, mul_tensor_60);  mul_tensor_53 = mul_tensor_60 = None
        sub_tensor_4: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_9);  sub_tensor_3 = unsqueeze_default_9 = None
        mul_tensor_61: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_15);  sub_tensor_4 = unsqueeze_default_15 = None
        mul_tensor_62: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_34, squeeze_dims_1);  sum_dim_int_list_34 = squeeze_dims_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, sum_dim_int_list_1, sum_dim_int_list_2, mul_tensor_2, mul_tensor_3, mul_tensor_4, sum_dim_int_list_3, sum_dim_int_list_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, sum_dim_int_list_5, sum_dim_int_list_6, mul_tensor_8, mul_tensor_9, mul_tensor_10, sum_dim_int_list_7, sum_dim_int_list_8, mul_tensor_11, mul_tensor_12, mul_tensor_13, sum_dim_int_list_9, sum_dim_int_list_10, mul_tensor_14, mul_tensor_15, mul_tensor_16, sum_dim_int_list_11, sum_dim_int_list_12, mul_tensor_17, mul_tensor_18, mul_tensor_19, sum_dim_int_list_13, sum_dim_int_list_14, mul_tensor_20, mul_tensor_21, mul_tensor_22, sum_dim_int_list_15, sum_dim_int_list_16, mul_tensor_23, mul_tensor_24, mul_tensor_25, sum_dim_int_list_17, sum_dim_int_list_18, mul_tensor_26, mul_tensor_27, mul_tensor_28, sum_dim_int_list_19, sum_dim_int_list_20, mul_tensor_29, mul_tensor_30, mul_tensor_31, sum_dim_int_list_21, sum_dim_int_list_22, mul_tensor_32, mul_tensor_33, mul_tensor_34, sum_dim_int_list_23, sum_dim_int_list_24, mul_tensor_35, mul_tensor_36, mul_tensor_37, sum_dim_int_list_25, sum_dim_int_list_26, mul_tensor_38, mul_tensor_39, mul_tensor_40, sum_dim_int_list_27, sum_dim_int_list_28, mul_tensor_41, mul_tensor_42, mul_tensor_43, sum_dim_int_list_29, sum_dim_int_list_30, mul_tensor_44, mul_tensor_45, mul_tensor_46, sum_dim_int_list_31, sum_dim_int_list_32, mul_tensor_47, mul_tensor_61, mul_tensor_62)



def make_inputs():
    return [
    torch.randn([4, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 48, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 48, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 48, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 48, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([4, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 20, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([4, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 20, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([4, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 20, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([4, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 10, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([4, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 10, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([4, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 6, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([4, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 6, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([4, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 4, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 32, 112, 112], [401408, 1, 3584, 32]),  # convolution
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 32, 112, 112], [401408, 1, 3584, 32]),  # getitem_335
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
