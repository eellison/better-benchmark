"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 65be000b769d
Shape hash: 659def5f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", permute_43: "f32[128, 1, 1, 640]", mul_92: "f32[128, 1, 1, 640]", div_15: "f32[128, 640, 7, 7]", mul_90: "f32[128, 2560, 7, 7]", div_13: "f32[128, 2560, 1, 1]", getitem_38: "f32[128, 2560, 7, 7]", mul_115: "f32[128, 2560, 7, 7]", permute_45: "f32[128, 7, 7, 640]", mul_86: "f32[128, 7, 7, 640]", permute_46: "f32[128, 640, 7, 7]", add_99: "f32[128, 640, 7, 7]", mul_84: "f32[128, 2560, 7, 7]", div_12: "f32[128, 2560, 1, 1]", getitem_47: "f32[128, 2560, 7, 7]", mul_137: "f32[128, 2560, 7, 7]", permute_47: "f32[128, 7, 7, 640]", mul_80: "f32[128, 7, 7, 640]", permute_48: "f32[128, 640, 7, 7]", add_105: "f32[128, 640, 7, 7]", permute_49: "f32[128, 14, 14, 320]", mul_78: "f32[128, 14, 14, 320]", permute_50: "f32[128, 320, 14, 14]", mul_76: "f32[128, 1280, 14, 14]", div_11: "f32[128, 1280, 1, 1]", getitem_59: "f32[128, 1280, 14, 14]", mul_166: "f32[128, 1280, 14, 14]", permute_51: "f32[128, 14, 14, 320]", mul_72: "f32[128, 14, 14, 320]", permute_52: "f32[128, 320, 14, 14]", add_111: "f32[128, 320, 14, 14]", mul_70: "f32[128, 1280, 14, 14]", div_10: "f32[128, 1280, 1, 1]", getitem_68: "f32[128, 1280, 14, 14]", mul_188: "f32[128, 1280, 14, 14]", permute_53: "f32[128, 14, 14, 320]", mul_66: "f32[128, 14, 14, 320]", permute_54: "f32[128, 320, 14, 14]", add_117: "f32[128, 320, 14, 14]", mul_64: "f32[128, 1280, 14, 14]", div_9: "f32[128, 1280, 1, 1]", getitem_77: "f32[128, 1280, 14, 14]", mul_210: "f32[128, 1280, 14, 14]", permute_55: "f32[128, 14, 14, 320]", mul_60: "f32[128, 14, 14, 320]", permute_56: "f32[128, 320, 14, 14]", add_123: "f32[128, 320, 14, 14]", mul_58: "f32[128, 1280, 14, 14]", div_8: "f32[128, 1280, 1, 1]", getitem_86: "f32[128, 1280, 14, 14]", mul_232: "f32[128, 1280, 14, 14]", permute_57: "f32[128, 14, 14, 320]", mul_54: "f32[128, 14, 14, 320]", permute_58: "f32[128, 320, 14, 14]", add_129: "f32[128, 320, 14, 14]", mul_52: "f32[128, 1280, 14, 14]", div_7: "f32[128, 1280, 1, 1]", getitem_95: "f32[128, 1280, 14, 14]", mul_254: "f32[128, 1280, 14, 14]", permute_59: "f32[128, 14, 14, 320]", mul_48: "f32[128, 14, 14, 320]", permute_60: "f32[128, 320, 14, 14]", add_135: "f32[128, 320, 14, 14]", mul_46: "f32[128, 1280, 14, 14]", div_6: "f32[128, 1280, 1, 1]", getitem_104: "f32[128, 1280, 14, 14]", mul_276: "f32[128, 1280, 14, 14]", permute_61: "f32[128, 14, 14, 320]", mul_42: "f32[128, 14, 14, 320]", permute_62: "f32[128, 320, 14, 14]", add_141: "f32[128, 320, 14, 14]", mul_40: "f32[128, 1280, 14, 14]", div_5: "f32[128, 1280, 1, 1]", getitem_113: "f32[128, 1280, 14, 14]", mul_298: "f32[128, 1280, 14, 14]", permute_63: "f32[128, 14, 14, 320]", mul_36: "f32[128, 14, 14, 320]", permute_64: "f32[128, 320, 14, 14]", add_147: "f32[128, 320, 14, 14]", mul_34: "f32[128, 1280, 14, 14]", div_4: "f32[128, 1280, 1, 1]", getitem_122: "f32[128, 1280, 14, 14]", mul_320: "f32[128, 1280, 14, 14]", permute_65: "f32[128, 14, 14, 320]", mul_30: "f32[128, 14, 14, 320]", permute_66: "f32[128, 320, 14, 14]", add_153: "f32[128, 320, 14, 14]", permute_67: "f32[128, 28, 28, 160]", mul_28: "f32[128, 28, 28, 160]", permute_68: "f32[128, 160, 28, 28]", mul_26: "f32[128, 640, 28, 28]", div_3: "f32[128, 640, 1, 1]", getitem_134: "f32[128, 640, 28, 28]", mul_349: "f32[128, 640, 28, 28]", permute_69: "f32[128, 28, 28, 160]", mul_22: "f32[128, 28, 28, 160]", permute_70: "f32[128, 160, 28, 28]", add_159: "f32[128, 160, 28, 28]", mul_20: "f32[128, 640, 28, 28]", div_2: "f32[128, 640, 1, 1]", getitem_143: "f32[128, 640, 28, 28]", mul_371: "f32[128, 640, 28, 28]", permute_71: "f32[128, 28, 28, 160]", mul_16: "f32[128, 28, 28, 160]", permute_72: "f32[128, 160, 28, 28]", add_165: "f32[128, 160, 28, 28]", permute_73: "f32[128, 56, 56, 80]", mul_14: "f32[128, 56, 56, 80]", permute_74: "f32[128, 80, 56, 56]", mul_12: "f32[128, 320, 56, 56]", div_1: "f32[128, 320, 1, 1]", getitem_155: "f32[128, 320, 56, 56]", mul_400: "f32[128, 320, 56, 56]", permute_75: "f32[128, 56, 56, 80]", mul_8: "f32[128, 56, 56, 80]", permute_76: "f32[128, 80, 56, 56]", add_171: "f32[128, 80, 56, 56]", mul_6: "f32[128, 320, 56, 56]", div: "f32[128, 320, 1, 1]", getitem_164: "f32[128, 320, 56, 56]", mul_422: "f32[128, 320, 56, 56]", permute_77: "f32[128, 56, 56, 80]", mul_2: "f32[128, 56, 56, 80]", permute_78: "f32[128, 80, 56, 56]", permute_79: "f32[128, 56, 56, 80]", mul: "f32[128, 56, 56, 80]", permute_80: "f32[128, 80, 56, 56]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(permute_43, mul_92);  mul_92 = None
        sum_dim_int_list_1: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1, 2]);  mul_tensor = None
        sum_dim_int_list_2: "f32[640]" = torch.ops.aten.sum.dim_IntList(permute_43, [0, 1, 2]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_3: "f32[640]" = torch.ops.aten.sum.dim_IntList(div_15, [0, 2, 3]);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_90, div_13);  mul_90 = div_13 = None
        mul_scalar: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Scalar(mul_tensor_1, 1);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_38, mul_scalar);  mul_scalar = None
        sum_dim_int_list_4: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_38, [0, 2, 3], True);  getitem_38 = None
        sum_dim_int_list_5: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3], True);  mul_tensor_2 = None
        reshape_default_1: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_1);  sum_dim_int_list_5 = _shape_param_1 = None
        reshape_default_2: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 2, 3]);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_3: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(permute_45, mul_86);  mul_86 = None
        sum_dim_int_list_7: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1, 2]);  mul_tensor_3 = None
        sum_dim_int_list_8: "f32[640]" = torch.ops.aten.sum.dim_IntList(permute_45, [0, 1, 2]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_9: "f32[640]" = torch.ops.aten.sum.dim_IntList(permute_46, [0, 2, 3]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_10: "f32[640]" = torch.ops.aten.sum.dim_IntList(add_99, [0, 2, 3]);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_4: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_84, div_12);  mul_84 = div_12 = None
        mul_scalar_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Scalar(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_47, mul_scalar_1);  mul_scalar_1 = None
        sum_dim_int_list_11: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_47, [0, 2, 3], True);  getitem_47 = None
        sum_dim_int_list_12: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3], True);  mul_tensor_5 = None
        reshape_default_3: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_3);  sum_dim_int_list_12 = _shape_param_3 = None
        reshape_default_4: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_4);  sum_dim_int_list_11 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_13: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_137, [0, 2, 3]);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_6: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(permute_47, mul_80);  mul_80 = None
        sum_dim_int_list_14: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2]);  mul_tensor_6 = None
        sum_dim_int_list_15: "f32[640]" = torch.ops.aten.sum.dim_IntList(permute_47, [0, 1, 2]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_16: "f32[640]" = torch.ops.aten.sum.dim_IntList(permute_48, [0, 2, 3]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_dim_int_list_17: "f32[640]" = torch.ops.aten.sum.dim_IntList(add_105, [0, 2, 3]);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_7: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_49, mul_78);  mul_78 = None
        sum_dim_int_list_18: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1, 2]);  mul_tensor_7 = None
        sum_dim_int_list_19: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_49, [0, 1, 2]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_20: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_50, [0, 2, 3]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_8: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, div_11);  mul_76 = div_11 = None
        mul_scalar_2: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_8, 1);  mul_tensor_8 = None
        mul_tensor_9: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_59, mul_scalar_2);  mul_scalar_2 = None
        sum_dim_int_list_21: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_59, [0, 2, 3], True);  getitem_59 = None
        sum_dim_int_list_22: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 2, 3], True);  mul_tensor_9 = None
        reshape_default_5: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_5);  sum_dim_int_list_22 = _shape_param_5 = None
        reshape_default_6: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_6);  sum_dim_int_list_21 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_23: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 2, 3]);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_10: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_51, mul_72);  mul_72 = None
        sum_dim_int_list_24: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1, 2]);  mul_tensor_10 = None
        sum_dim_int_list_25: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_51, [0, 1, 2]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_26: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_52, [0, 2, 3]);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_27: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 2, 3]);  add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_11: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, div_10);  mul_70 = div_10 = None
        mul_scalar_3: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_11, 1);  mul_tensor_11 = None
        mul_tensor_12: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_68, mul_scalar_3);  mul_scalar_3 = None
        sum_dim_int_list_28: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_68, [0, 2, 3], True);  getitem_68 = None
        sum_dim_int_list_29: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3], True);  mul_tensor_12 = None
        reshape_default_7: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_7);  sum_dim_int_list_29 = _shape_param_7 = None
        reshape_default_8: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_8);  sum_dim_int_list_28 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_30: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3]);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_13: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_53, mul_66);  mul_66 = None
        sum_dim_int_list_31: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1, 2]);  mul_tensor_13 = None
        sum_dim_int_list_32: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_53, [0, 1, 2]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_33: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_54, [0, 2, 3]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_34: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 2, 3]);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_14: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_64, div_9);  mul_64 = div_9 = None
        mul_scalar_4: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_14, 1);  mul_tensor_14 = None
        mul_tensor_15: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_77, mul_scalar_4);  mul_scalar_4 = None
        sum_dim_int_list_35: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_77, [0, 2, 3], True);  getitem_77 = None
        sum_dim_int_list_36: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 2, 3], True);  mul_tensor_15 = None
        reshape_default_9: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_9);  sum_dim_int_list_36 = _shape_param_9 = None
        reshape_default_10: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_10);  sum_dim_int_list_35 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_37: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 2, 3]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_16: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_55, mul_60);  mul_60 = None
        sum_dim_int_list_38: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1, 2]);  mul_tensor_16 = None
        sum_dim_int_list_39: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_55, [0, 1, 2]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_40: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_56, [0, 2, 3]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_41: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_17: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_58, div_8);  mul_58 = div_8 = None
        mul_scalar_5: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_17, 1);  mul_tensor_17 = None
        mul_tensor_18: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_86, mul_scalar_5);  mul_scalar_5 = None
        sum_dim_int_list_42: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_86, [0, 2, 3], True);  getitem_86 = None
        sum_dim_int_list_43: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 2, 3], True);  mul_tensor_18 = None
        reshape_default_11: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_11);  sum_dim_int_list_43 = _shape_param_11 = None
        reshape_default_12: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_12);  sum_dim_int_list_42 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_44: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 2, 3]);  mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_19: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_57, mul_54);  mul_54 = None
        sum_dim_int_list_45: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1, 2]);  mul_tensor_19 = None
        sum_dim_int_list_46: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_57, [0, 1, 2]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_47: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_58, [0, 2, 3]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_48: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 2, 3]);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_20: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_52, div_7);  mul_52 = div_7 = None
        mul_scalar_6: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_20, 1);  mul_tensor_20 = None
        mul_tensor_21: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_95, mul_scalar_6);  mul_scalar_6 = None
        sum_dim_int_list_49: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_95, [0, 2, 3], True);  getitem_95 = None
        sum_dim_int_list_50: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 2, 3], True);  mul_tensor_21 = None
        reshape_default_13: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_13);  sum_dim_int_list_50 = _shape_param_13 = None
        reshape_default_14: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_14);  sum_dim_int_list_49 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_51: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3]);  mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_22: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_59, mul_48);  mul_48 = None
        sum_dim_int_list_52: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1, 2]);  mul_tensor_22 = None
        sum_dim_int_list_53: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_59, [0, 1, 2]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_54: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_60, [0, 2, 3]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_55: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 2, 3]);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_23: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_46, div_6);  mul_46 = div_6 = None
        mul_scalar_7: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_23, 1);  mul_tensor_23 = None
        mul_tensor_24: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_104, mul_scalar_7);  mul_scalar_7 = None
        sum_dim_int_list_56: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_104, [0, 2, 3], True);  getitem_104 = None
        sum_dim_int_list_57: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 2, 3], True);  mul_tensor_24 = None
        reshape_default_15: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_15);  sum_dim_int_list_57 = _shape_param_15 = None
        reshape_default_16: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_16);  sum_dim_int_list_56 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_58: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 2, 3]);  mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_25: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_61, mul_42);  mul_42 = None
        sum_dim_int_list_59: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1, 2]);  mul_tensor_25 = None
        sum_dim_int_list_60: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_61, [0, 1, 2]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_61: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_62, [0, 2, 3]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_62: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 2, 3]);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_26: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_40, div_5);  mul_40 = div_5 = None
        mul_scalar_8: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_26, 1);  mul_tensor_26 = None
        mul_tensor_27: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_113, mul_scalar_8);  mul_scalar_8 = None
        sum_dim_int_list_63: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3], True);  getitem_113 = None
        sum_dim_int_list_64: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 2, 3], True);  mul_tensor_27 = None
        reshape_default_17: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_17);  sum_dim_int_list_64 = _shape_param_17 = None
        reshape_default_18: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_18);  sum_dim_int_list_63 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_65: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3]);  mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_28: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_63, mul_36);  mul_36 = None
        sum_dim_int_list_66: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1, 2]);  mul_tensor_28 = None
        sum_dim_int_list_67: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_63, [0, 1, 2]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_68: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_64, [0, 2, 3]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_69: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3]);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_29: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_34, div_4);  mul_34 = div_4 = None
        mul_scalar_9: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Scalar(mul_tensor_29, 1);  mul_tensor_29 = None
        mul_tensor_30: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_122, mul_scalar_9);  mul_scalar_9 = None
        sum_dim_int_list_70: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_122, [0, 2, 3], True);  getitem_122 = None
        sum_dim_int_list_71: "f32[1, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 2, 3], True);  mul_tensor_30 = None
        reshape_default_19: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_19);  sum_dim_int_list_71 = _shape_param_19 = None
        reshape_default_20: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_20);  sum_dim_int_list_70 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_72: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 2, 3]);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_31: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_65, mul_30);  mul_30 = None
        sum_dim_int_list_73: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1, 2]);  mul_tensor_31 = None
        sum_dim_int_list_74: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_65, [0, 1, 2]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_75: "f32[320]" = torch.ops.aten.sum.dim_IntList(permute_66, [0, 2, 3]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_dim_int_list_76: "f32[320]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3]);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_32: "f32[128, 28, 28, 160]" = torch.ops.aten.mul.Tensor(permute_67, mul_28);  mul_28 = None
        sum_dim_int_list_77: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1, 2]);  mul_tensor_32 = None
        sum_dim_int_list_78: "f32[160]" = torch.ops.aten.sum.dim_IntList(permute_67, [0, 1, 2]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_79: "f32[160]" = torch.ops.aten.sum.dim_IntList(permute_68, [0, 2, 3]);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_33: "f32[128, 640, 28, 28]" = torch.ops.aten.mul.Tensor(mul_26, div_3);  mul_26 = div_3 = None
        mul_scalar_10: "f32[128, 640, 28, 28]" = torch.ops.aten.mul.Scalar(mul_tensor_33, 1);  mul_tensor_33 = None
        mul_tensor_34: "f32[128, 640, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_134, mul_scalar_10);  mul_scalar_10 = None
        sum_dim_int_list_80: "f32[1, 640, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_134, [0, 2, 3], True);  getitem_134 = None
        sum_dim_int_list_81: "f32[1, 640, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 2, 3], True);  mul_tensor_34 = None
        reshape_default_21: "f32[640]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_21);  sum_dim_int_list_81 = _shape_param_21 = None
        reshape_default_22: "f32[640]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_22);  sum_dim_int_list_80 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_82: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 2, 3]);  mul_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_35: "f32[128, 28, 28, 160]" = torch.ops.aten.mul.Tensor(permute_69, mul_22);  mul_22 = None
        sum_dim_int_list_83: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1, 2]);  mul_tensor_35 = None
        sum_dim_int_list_84: "f32[160]" = torch.ops.aten.sum.dim_IntList(permute_69, [0, 1, 2]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_85: "f32[160]" = torch.ops.aten.sum.dim_IntList(permute_70, [0, 2, 3]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_86: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 2, 3]);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_36: "f32[128, 640, 28, 28]" = torch.ops.aten.mul.Tensor(mul_20, div_2);  mul_20 = div_2 = None
        mul_scalar_11: "f32[128, 640, 28, 28]" = torch.ops.aten.mul.Scalar(mul_tensor_36, 1);  mul_tensor_36 = None
        mul_tensor_37: "f32[128, 640, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_143, mul_scalar_11);  mul_scalar_11 = None
        sum_dim_int_list_87: "f32[1, 640, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_143, [0, 2, 3], True);  getitem_143 = None
        sum_dim_int_list_88: "f32[1, 640, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 2, 3], True);  mul_tensor_37 = None
        reshape_default_23: "f32[640]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_23);  sum_dim_int_list_88 = _shape_param_23 = None
        reshape_default_24: "f32[640]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_24);  sum_dim_int_list_87 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_89: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_38: "f32[128, 28, 28, 160]" = torch.ops.aten.mul.Tensor(permute_71, mul_16);  mul_16 = None
        sum_dim_int_list_90: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1, 2]);  mul_tensor_38 = None
        sum_dim_int_list_91: "f32[160]" = torch.ops.aten.sum.dim_IntList(permute_71, [0, 1, 2]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_92: "f32[160]" = torch.ops.aten.sum.dim_IntList(permute_72, [0, 2, 3]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_dim_int_list_93: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 2, 3]);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_39: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_73, mul_14);  mul_14 = None
        sum_dim_int_list_94: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1, 2]);  mul_tensor_39 = None
        sum_dim_int_list_95: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_73, [0, 1, 2]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_96: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_74, [0, 2, 3]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_40: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_12, div_1);  mul_12 = div_1 = None
        mul_scalar_12: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Scalar(mul_tensor_40, 1);  mul_tensor_40 = None
        mul_tensor_41: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_155, mul_scalar_12);  mul_scalar_12 = None
        sum_dim_int_list_97: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_155, [0, 2, 3], True);  getitem_155 = None
        sum_dim_int_list_98: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 2, 3], True);  mul_tensor_41 = None
        reshape_default_25: "f32[320]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_25);  sum_dim_int_list_98 = _shape_param_25 = None
        reshape_default_26: "f32[320]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_26);  sum_dim_int_list_97 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_99: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_42: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_75, mul_8);  mul_8 = None
        sum_dim_int_list_100: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1, 2]);  mul_tensor_42 = None
        sum_dim_int_list_101: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_75, [0, 1, 2]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_102: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_76, [0, 2, 3]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_dim_int_list_103: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 2, 3]);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_43: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_6, div);  mul_6 = div = None
        mul_scalar_13: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Scalar(mul_tensor_43, 1);  mul_tensor_43 = None
        mul_tensor_44: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_164, mul_scalar_13);  mul_scalar_13 = None
        sum_dim_int_list_104: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_164, [0, 2, 3], True);  getitem_164 = None
        sum_dim_int_list_105: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 2, 3], True);  mul_tensor_44 = None
        reshape_default_27: "f32[320]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, _shape_param_27);  sum_dim_int_list_105 = _shape_param_27 = None
        reshape_default_28: "f32[320]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_28);  sum_dim_int_list_104 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_106: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 2, 3]);  mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_45: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_77, mul_2);  mul_2 = None
        sum_dim_int_list_107: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1, 2]);  mul_tensor_45 = None
        sum_dim_int_list_108: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_77, [0, 1, 2]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_dim_int_list_109: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_78, [0, 2, 3]);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_46: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_79, mul);  mul = None
        sum_dim_int_list_110: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1, 2]);  mul_tensor_46 = None
        sum_dim_int_list_111: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_79, [0, 1, 2]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:608 in forward_features, code: x = self.stem(x)
        sum_dim_int_list_112: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_80, [0, 2, 3]);  permute_80 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, sum_dim_int_list_3, reshape_default_1, reshape_default_2, sum_dim_int_list_6, sum_dim_int_list_7, sum_dim_int_list_8, sum_dim_int_list_9, sum_dim_int_list_10, reshape_default_3, reshape_default_4, sum_dim_int_list_13, sum_dim_int_list_14, sum_dim_int_list_15, sum_dim_int_list_16, sum_dim_int_list_17, sum_dim_int_list_18, sum_dim_int_list_19, sum_dim_int_list_20, reshape_default_5, reshape_default_6, sum_dim_int_list_23, sum_dim_int_list_24, sum_dim_int_list_25, sum_dim_int_list_26, sum_dim_int_list_27, reshape_default_7, reshape_default_8, sum_dim_int_list_30, sum_dim_int_list_31, sum_dim_int_list_32, sum_dim_int_list_33, sum_dim_int_list_34, reshape_default_9, reshape_default_10, sum_dim_int_list_37, sum_dim_int_list_38, sum_dim_int_list_39, sum_dim_int_list_40, sum_dim_int_list_41, reshape_default_11, reshape_default_12, sum_dim_int_list_44, sum_dim_int_list_45, sum_dim_int_list_46, sum_dim_int_list_47, sum_dim_int_list_48, reshape_default_13, reshape_default_14, sum_dim_int_list_51, sum_dim_int_list_52, sum_dim_int_list_53, sum_dim_int_list_54, sum_dim_int_list_55, reshape_default_15, reshape_default_16, sum_dim_int_list_58, sum_dim_int_list_59, sum_dim_int_list_60, sum_dim_int_list_61, sum_dim_int_list_62, reshape_default_17, reshape_default_18, sum_dim_int_list_65, sum_dim_int_list_66, sum_dim_int_list_67, sum_dim_int_list_68, sum_dim_int_list_69, reshape_default_19, reshape_default_20, sum_dim_int_list_72, sum_dim_int_list_73, sum_dim_int_list_74, sum_dim_int_list_75, sum_dim_int_list_76, sum_dim_int_list_77, sum_dim_int_list_78, sum_dim_int_list_79, reshape_default_21, reshape_default_22, sum_dim_int_list_82, sum_dim_int_list_83, sum_dim_int_list_84, sum_dim_int_list_85, sum_dim_int_list_86, reshape_default_23, reshape_default_24, sum_dim_int_list_89, sum_dim_int_list_90, sum_dim_int_list_91, sum_dim_int_list_92, sum_dim_int_list_93, sum_dim_int_list_94, sum_dim_int_list_95, sum_dim_int_list_96, reshape_default_25, reshape_default_26, sum_dim_int_list_99, sum_dim_int_list_100, sum_dim_int_list_101, sum_dim_int_list_102, sum_dim_int_list_103, reshape_default_27, reshape_default_28, sum_dim_int_list_106, sum_dim_int_list_107, sum_dim_int_list_108, sum_dim_int_list_109, sum_dim_int_list_110, sum_dim_int_list_111, sum_dim_int_list_112)


def _default_make_inputs():
    return [
    torch.randn([128, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1, 1, 640], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1, 1, 640], dtype=torch.float32, device='cuda'),
    torch.randn([128, 640, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 2560, 7, 7], [125440, 1, 17920, 2560]),  # mul_90
    torch.randn([128, 2560, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 2560, 7, 7], [125440, 1, 17920, 2560]),  # getitem_38
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 2560, 7, 7], [125440, 1, 17920, 2560]),  # mul_115
    torch.randn([128, 7, 7, 640], dtype=torch.float32, device='cuda'),
    torch.randn([128, 7, 7, 640], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 640, 7, 7], [31360, 1, 640, 4480]),  # permute_46
    torch.randn([128, 640, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 2560, 7, 7], [125440, 1, 17920, 2560]),  # mul_84
    torch.randn([128, 2560, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 2560, 7, 7], [125440, 1, 17920, 2560]),  # getitem_47
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 2560, 7, 7], [125440, 1, 17920, 2560]),  # mul_137
    torch.randn([128, 7, 7, 640], dtype=torch.float32, device='cuda'),
    torch.randn([128, 7, 7, 640], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 640, 7, 7], [31360, 1, 640, 4480]),  # permute_48
    torch.randn([128, 640, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 14, 14, 320], [62720, 1, 4480, 14]),  # mul_78
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_50
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_76
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_59
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_166
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_52
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_111
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_70
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_68
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_188
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_54
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_117
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_64
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_77
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_210
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_56
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_123
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_58
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_86
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_232
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_58
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_129
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_52
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_95
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_254
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_60
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_135
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_46
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_104
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_276
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_62
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_141
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_40
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_113
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_298
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_64
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_147
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_34
    torch.randn([128, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # getitem_122
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 1280, 14, 14], [250880, 1, 17920, 1280]),  # mul_320
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn([128, 14, 14, 320], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # permute_66
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 320, 4480]),  # add_153
    torch.randn([128, 28, 28, 160], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 28, 28, 160], [125440, 1, 4480, 28]),  # mul_28
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 160, 28, 28], [125440, 1, 160, 4480]),  # permute_68
    torch.randn(64225280, dtype=torch.float32, device='cuda').as_strided([128, 640, 28, 28], [501760, 1, 17920, 640]),  # mul_26
    torch.randn([128, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(64225280, dtype=torch.float32, device='cuda').as_strided([128, 640, 28, 28], [501760, 1, 17920, 640]),  # getitem_134
    torch.randn(64225280, dtype=torch.float32, device='cuda').as_strided([128, 640, 28, 28], [501760, 1, 17920, 640]),  # mul_349
    torch.randn([128, 28, 28, 160], dtype=torch.float32, device='cuda'),
    torch.randn([128, 28, 28, 160], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 160, 28, 28], [125440, 1, 160, 4480]),  # permute_70
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 160, 28, 28], [125440, 1, 160, 4480]),  # add_159
    torch.randn(64225280, dtype=torch.float32, device='cuda').as_strided([128, 640, 28, 28], [501760, 1, 17920, 640]),  # mul_20
    torch.randn([128, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(64225280, dtype=torch.float32, device='cuda').as_strided([128, 640, 28, 28], [501760, 1, 17920, 640]),  # getitem_143
    torch.randn(64225280, dtype=torch.float32, device='cuda').as_strided([128, 640, 28, 28], [501760, 1, 17920, 640]),  # mul_371
    torch.randn([128, 28, 28, 160], dtype=torch.float32, device='cuda'),
    torch.randn([128, 28, 28, 160], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 160, 28, 28], [125440, 1, 160, 4480]),  # permute_72
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([128, 160, 28, 28], [125440, 1, 160, 4480]),  # add_165
    torch.randn([128, 56, 56, 80], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 56, 56, 80], [250880, 1, 4480, 56]),  # mul_14
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 80, 56, 56], [250880, 1, 80, 4480]),  # permute_74
    torch.randn(128450560, dtype=torch.float32, device='cuda').as_strided([128, 320, 56, 56], [1003520, 1, 17920, 320]),  # mul_12
    torch.randn([128, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(128450560, dtype=torch.float32, device='cuda').as_strided([128, 320, 56, 56], [1003520, 1, 17920, 320]),  # getitem_155
    torch.randn(128450560, dtype=torch.float32, device='cuda').as_strided([128, 320, 56, 56], [1003520, 1, 17920, 320]),  # mul_400
    torch.randn([128, 56, 56, 80], dtype=torch.float32, device='cuda'),
    torch.randn([128, 56, 56, 80], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 80, 56, 56], [250880, 1, 80, 4480]),  # permute_76
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 80, 56, 56], [250880, 1, 80, 4480]),  # add_171
    torch.randn(128450560, dtype=torch.float32, device='cuda').as_strided([128, 320, 56, 56], [1003520, 1, 17920, 320]),  # mul_6
    torch.randn([128, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(128450560, dtype=torch.float32, device='cuda').as_strided([128, 320, 56, 56], [1003520, 1, 17920, 320]),  # getitem_164
    torch.randn(128450560, dtype=torch.float32, device='cuda').as_strided([128, 320, 56, 56], [1003520, 1, 17920, 320]),  # mul_422
    torch.randn([128, 56, 56, 80], dtype=torch.float32, device='cuda'),
    torch.randn([128, 56, 56, 80], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 80, 56, 56], [250880, 1, 80, 4480]),  # permute_78
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 56, 56, 80], [250880, 80, 4480, 1]),  # permute_79
    torch.randn([128, 56, 56, 80], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 80, 56, 56], [250880, 1, 80, 4480]),  # permute_80
    [1000],  # _shape_param_0
    [2560],  # _shape_param_1
    [2560],  # _shape_param_2
    [2560],  # _shape_param_3
    [2560],  # _shape_param_4
    [1280],  # _shape_param_5
    [1280],  # _shape_param_6
    [1280],  # _shape_param_7
    [1280],  # _shape_param_8
    [1280],  # _shape_param_9
    [1280],  # _shape_param_10
    [1280],  # _shape_param_11
    [1280],  # _shape_param_12
    [1280],  # _shape_param_13
    [1280],  # _shape_param_14
    [1280],  # _shape_param_15
    [1280],  # _shape_param_16
    [1280],  # _shape_param_17
    [1280],  # _shape_param_18
    [1280],  # _shape_param_19
    [1280],  # _shape_param_20
    [640],  # _shape_param_21
    [640],  # _shape_param_22
    [640],  # _shape_param_23
    [640],  # _shape_param_24
    [320],  # _shape_param_25
    [320],  # _shape_param_26
    [320],  # _shape_param_27
    [320],  # _shape_param_28
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
