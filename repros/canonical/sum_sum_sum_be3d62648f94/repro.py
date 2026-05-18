"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_training
Pattern hash: be3d62648f94
Shape hash: e6445ab3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[32, 1000]", _shape_param_0, where_1: "f32[32, 1280, 1, 1]", sum_4: "f32[960]", squeeze_136: "f32[960]", sum_6: "f32[160]", squeeze_133: "f32[160]", where_4: "f32[32, 960, 1, 1]", where_5: "f32[32, 240, 1, 1]", sum_11: "f32[960]", squeeze_130: "f32[960]", sum_13: "f32[960]", squeeze_127: "f32[960]", sum_15: "f32[160]", squeeze_124: "f32[160]", where_10: "f32[32, 960, 1, 1]", where_11: "f32[32, 240, 1, 1]", sum_20: "f32[960]", squeeze_121: "f32[960]", sum_22: "f32[960]", squeeze_118: "f32[960]", sum_24: "f32[160]", squeeze_115: "f32[160]", where_16: "f32[32, 672, 1, 1]", where_17: "f32[32, 168, 1, 1]", sum_29: "f32[672]", squeeze_112: "f32[672]", sum_31: "f32[672]", squeeze_109: "f32[672]", sum_33: "f32[112]", squeeze_106: "f32[112]", where_22: "f32[32, 672, 1, 1]", where_23: "f32[32, 168, 1, 1]", sum_38: "f32[672]", squeeze_103: "f32[672]", sum_40: "f32[672]", squeeze_100: "f32[672]", sum_42: "f32[112]", squeeze_97: "f32[112]", where_28: "f32[32, 480, 1, 1]", where_29: "f32[32, 120, 1, 1]", sum_47: "f32[480]", squeeze_94: "f32[480]", sum_49: "f32[480]", squeeze_91: "f32[480]", sum_51: "f32[80]", squeeze_88: "f32[80]", sum_53: "f32[184]", squeeze_85: "f32[184]", sum_55: "f32[184]", squeeze_82: "f32[184]", sum_57: "f32[80]", squeeze_79: "f32[80]", sum_59: "f32[184]", squeeze_76: "f32[184]", sum_61: "f32[184]", squeeze_73: "f32[184]", sum_63: "f32[80]", squeeze_70: "f32[80]", sum_65: "f32[200]", squeeze_67: "f32[200]", sum_67: "f32[200]", squeeze_64: "f32[200]", sum_69: "f32[80]", squeeze_61: "f32[80]", sum_71: "f32[240]", squeeze_58: "f32[240]", sum_73: "f32[240]", squeeze_55: "f32[240]", sum_75: "f32[40]", squeeze_52: "f32[40]", where_50: "f32[32, 120, 1, 1]", where_51: "f32[32, 32, 1, 1]", sum_80: "f32[120]", squeeze_49: "f32[120]", sum_82: "f32[120]", squeeze_46: "f32[120]", sum_84: "f32[40]", squeeze_43: "f32[40]", where_54: "f32[32, 120, 1, 1]", where_55: "f32[32, 32, 1, 1]", sum_89: "f32[120]", squeeze_40: "f32[120]", sum_91: "f32[120]", squeeze_37: "f32[120]", sum_93: "f32[40]", squeeze_34: "f32[40]", where_58: "f32[32, 72, 1, 1]", where_59: "f32[32, 24, 1, 1]", sum_98: "f32[72]", squeeze_31: "f32[72]", sum_100: "f32[72]", squeeze_28: "f32[72]", sum_102: "f32[24]", squeeze_25: "f32[24]", sum_104: "f32[72]", squeeze_22: "f32[72]", sum_106: "f32[72]", squeeze_19: "f32[72]", sum_108: "f32[24]", squeeze_16: "f32[24]", sum_110: "f32[64]", squeeze_13: "f32[64]", sum_112: "f32[64]", squeeze_10: "f32[64]", sum_114: "f32[16]", squeeze_7: "f32[16]", sum_116: "f32[16]", squeeze_4: "f32[16]", getitem_269: "f32[32, 16, 112, 112]", getitem_275: "f32[32, 16, 112, 112]", convolution: "f32[32, 16, 112, 112]", getitem_1: "f32[1, 16, 1, 1]", rsqrt: "f32[1, 16, 1, 1]", primals_6: "f32[16]", primals_7: "f32[16]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_default: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:323 in forward_head, code: x = self.conv_head(x)
        sum_dim_int_list_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3]);  where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_136);  sum_4 = squeeze_136 = None
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_133);  sum_6 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_2: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3]);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_3: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3]);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_2: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_130);  sum_11 = squeeze_130 = None
        mul_tensor_3: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_127);  sum_13 = squeeze_127 = None
        mul_tensor_4: "f32[160]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_124);  sum_15 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3]);  where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_5: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3]);  where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_5: "f32[960]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_121);  sum_20 = squeeze_121 = None
        mul_tensor_6: "f32[960]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_118);  sum_22 = squeeze_118 = None
        mul_tensor_7: "f32[160]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_115);  sum_24 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_6: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3]);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_7: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3]);  where_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_8: "f32[672]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_112);  sum_29 = squeeze_112 = None
        mul_tensor_9: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_109);  sum_31 = squeeze_109 = None
        mul_tensor_10: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_106);  sum_33 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_8: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3]);  where_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_9: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3]);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_11: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_103);  sum_38 = squeeze_103 = None
        mul_tensor_12: "f32[672]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_100);  sum_40 = squeeze_100 = None
        mul_tensor_13: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_97);  sum_42 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_10: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3]);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_11: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3]);  where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_14: "f32[480]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_94);  sum_47 = squeeze_94 = None
        mul_tensor_15: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_91);  sum_49 = squeeze_91 = None
        mul_tensor_16: "f32[80]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_88);  sum_51 = squeeze_88 = None
        mul_tensor_17: "f32[184]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_85);  sum_53 = squeeze_85 = None
        mul_tensor_18: "f32[184]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_82);  sum_55 = squeeze_82 = None
        mul_tensor_19: "f32[80]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_79);  sum_57 = squeeze_79 = None
        mul_tensor_20: "f32[184]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_76);  sum_59 = squeeze_76 = None
        mul_tensor_21: "f32[184]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_73);  sum_61 = squeeze_73 = None
        mul_tensor_22: "f32[80]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_70);  sum_63 = squeeze_70 = None
        mul_tensor_23: "f32[200]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_67);  sum_65 = squeeze_67 = None
        mul_tensor_24: "f32[200]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_64);  sum_67 = squeeze_64 = None
        mul_tensor_25: "f32[80]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_61);  sum_69 = squeeze_61 = None
        mul_tensor_26: "f32[240]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_58);  sum_71 = squeeze_58 = None
        mul_tensor_27: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_55);  sum_73 = squeeze_55 = None
        mul_tensor_28: "f32[40]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_52);  sum_75 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_12: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3]);  where_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_13: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3]);  where_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_29: "f32[120]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_49);  sum_80 = squeeze_49 = None
        mul_tensor_30: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_46);  sum_82 = squeeze_46 = None
        mul_tensor_31: "f32[40]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_43);  sum_84 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_14: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3]);  where_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_15: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3]);  where_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_32: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_40);  sum_89 = squeeze_40 = None
        mul_tensor_33: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_37);  sum_91 = squeeze_37 = None
        mul_tensor_34: "f32[40]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_34);  sum_93 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_16: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3]);  where_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_17: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3]);  where_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_35: "f32[72]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_31);  sum_98 = squeeze_31 = None
        mul_tensor_36: "f32[72]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_28);  sum_100 = squeeze_28 = None
        mul_tensor_37: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_25);  sum_102 = squeeze_25 = None
        mul_tensor_38: "f32[72]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_22);  sum_104 = squeeze_22 = None
        mul_tensor_39: "f32[72]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_19);  sum_106 = squeeze_19 = None
        mul_tensor_40: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_16);  sum_108 = squeeze_16 = None
        mul_tensor_41: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_13);  sum_110 = squeeze_13 = None
        mul_tensor_42: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_10);  sum_112 = squeeze_10 = None
        mul_tensor_43: "f32[16]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_7);  sum_114 = squeeze_7 = None
        mul_tensor_44: "f32[16]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_4);  sum_116 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        add_tensor: "f32[32, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_269, getitem_275);  getitem_269 = getitem_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_tensor_45: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_default_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_46: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_45, unsqueeze_default_1);  mul_tensor_45 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_default_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_46, unsqueeze_default_3);  mul_tensor_46 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[32, 16, 112, 112]" = torch.ops.aten.le.Scalar(add_tensor_1, -3)
        lt_scalar: "b8[32, 16, 112, 112]" = torch.ops.aten.lt.Scalar(add_tensor_1, 3)
        div_tensor: "f32[32, 16, 112, 112]" = torch.ops.aten.div.Tensor(add_tensor_1, 3);  add_tensor_1 = None
        add_tensor_2: "f32[32, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_47: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_tensor, add_tensor_2);  add_tensor_2 = None
        where_self: "f32[32, 16, 112, 112]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_47, add_tensor);  lt_scalar = mul_tensor_47 = add_tensor = None
        where_self_1: "f32[32, 16, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_default_4: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list_18: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_1: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_default_6);  convolution = unsqueeze_default_6 = None
        mul_tensor_48: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_1)
        sum_dim_int_list_19: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 2, 3]);  mul_tensor_48 = None
        mul_tensor_49: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_18, 2.4912308673469386e-06);  sum_dim_int_list_18 = None
        unsqueeze_default_7: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_49, 0);  mul_tensor_49 = None
        unsqueeze_default_8: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_50: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_19, 2.4912308673469386e-06)
        squeeze_dims_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor_51: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_52: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        unsqueeze_default_10: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_52, 0);  mul_tensor_52 = None
        unsqueeze_default_11: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_53: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_6);  primals_6 = None
        unsqueeze_default_13: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_53, 0);  mul_tensor_53 = None
        unsqueeze_default_14: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_54: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_54);  where_self_1 = mul_tensor_54 = None
        sub_tensor_3: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_55: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_56: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_19, squeeze_dims_1);  sum_dim_int_list_19 = squeeze_dims_1 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, mul_tensor, mul_tensor_1, sum_dim_int_list_2, sum_dim_int_list_3, mul_tensor_2, mul_tensor_3, mul_tensor_4, sum_dim_int_list_4, sum_dim_int_list_5, mul_tensor_5, mul_tensor_6, mul_tensor_7, sum_dim_int_list_6, sum_dim_int_list_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, sum_dim_int_list_8, sum_dim_int_list_9, mul_tensor_11, mul_tensor_12, mul_tensor_13, sum_dim_int_list_10, sum_dim_int_list_11, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, sum_dim_int_list_12, sum_dim_int_list_13, mul_tensor_29, mul_tensor_30, mul_tensor_31, sum_dim_int_list_14, sum_dim_int_list_15, mul_tensor_32, mul_tensor_33, mul_tensor_34, sum_dim_int_list_16, sum_dim_int_list_17, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_55, mul_tensor_56)


def _default_make_inputs():
    return [
    torch.randn([32, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    torch.randn([32, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([32, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([32, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([32, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 168, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([32, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 168, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([32, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([32, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([32, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([32, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # getitem_269
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # getitem_275
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # convolution
    torch.randn([1, 16, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
