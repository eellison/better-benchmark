"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: ee919cdda8d3
Shape hash: b0f872d7
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[32, 1000]", _shape_param_0, where: "f32[32, 1280, 1, 1]", sum_4: "f32[960]", squeeze_238: "f32[960]", sum_6: "f32[80]", squeeze_235: "f32[80]", sum_8: "f32[80]", squeeze_232: "f32[80]", where_2: "f32[32, 960, 1, 1]", where_3: "f32[32, 240, 1, 1]", sum_13: "f32[480]", squeeze_229: "f32[480]", sum_15: "f32[480]", squeeze_226: "f32[480]", sum_17: "f32[80]", squeeze_223: "f32[80]", sum_19: "f32[80]", squeeze_220: "f32[80]", sum_21: "f32[480]", squeeze_217: "f32[480]", sum_23: "f32[480]", squeeze_214: "f32[480]", sum_25: "f32[80]", squeeze_211: "f32[80]", sum_27: "f32[80]", squeeze_208: "f32[80]", where_8: "f32[32, 960, 1, 1]", where_9: "f32[32, 240, 1, 1]", sum_32: "f32[480]", squeeze_205: "f32[480]", sum_34: "f32[480]", squeeze_202: "f32[480]", sum_36: "f32[80]", squeeze_199: "f32[80]", sum_38: "f32[80]", squeeze_196: "f32[80]", sum_40: "f32[480]", squeeze_193: "f32[480]", sum_42: "f32[480]", squeeze_190: "f32[480]", sum_44: "f32[160]", squeeze_187: "f32[160]", sum_46: "f32[112]", squeeze_184: "f32[112]", sum_48: "f32[80]", squeeze_181: "f32[80]", sum_50: "f32[80]", squeeze_178: "f32[80]", where_14: "f32[32, 672, 1, 1]", where_15: "f32[32, 168, 1, 1]", sum_55: "f32[672]", squeeze_175: "f32[672]", sum_57: "f32[336]", squeeze_172: "f32[336]", sum_59: "f32[336]", squeeze_169: "f32[336]", sum_61: "f32[56]", squeeze_166: "f32[56]", sum_63: "f32[56]", squeeze_163: "f32[56]", where_18: "f32[32, 672, 1, 1]", where_19: "f32[32, 168, 1, 1]", sum_68: "f32[336]", squeeze_160: "f32[336]", sum_70: "f32[336]", squeeze_157: "f32[336]", sum_72: "f32[112]", squeeze_154: "f32[112]", sum_74: "f32[80]", squeeze_151: "f32[80]", sum_76: "f32[56]", squeeze_148: "f32[56]", sum_78: "f32[56]", squeeze_145: "f32[56]", where_22: "f32[32, 480, 1, 1]", where_23: "f32[32, 120, 1, 1]", sum_83: "f32[240]", squeeze_142: "f32[240]", sum_85: "f32[240]", squeeze_139: "f32[240]", sum_87: "f32[40]", squeeze_136: "f32[40]", sum_89: "f32[40]", squeeze_133: "f32[40]", sum_91: "f32[92]", squeeze_130: "f32[92]", sum_93: "f32[92]", squeeze_127: "f32[92]", sum_95: "f32[40]", squeeze_124: "f32[40]", sum_97: "f32[40]", squeeze_121: "f32[40]", sum_99: "f32[92]", squeeze_118: "f32[92]", sum_101: "f32[92]", squeeze_115: "f32[92]", sum_103: "f32[40]", squeeze_112: "f32[40]", sum_105: "f32[40]", squeeze_109: "f32[40]", sum_107: "f32[100]", squeeze_106: "f32[100]", sum_109: "f32[100]", squeeze_103: "f32[100]", sum_111: "f32[80]", squeeze_100: "f32[80]", sum_113: "f32[40]", squeeze_97: "f32[40]", sum_115: "f32[40]", squeeze_94: "f32[40]", sum_117: "f32[40]", squeeze_91: "f32[40]", sum_119: "f32[240]", squeeze_88: "f32[240]", sum_121: "f32[120]", squeeze_85: "f32[120]", sum_123: "f32[120]", squeeze_82: "f32[120]", sum_125: "f32[20]", squeeze_79: "f32[20]", sum_127: "f32[20]", squeeze_76: "f32[20]", where_34: "f32[32, 120, 1, 1]", where_35: "f32[32, 32, 1, 1]", sum_132: "f32[60]", squeeze_73: "f32[60]", sum_134: "f32[60]", squeeze_70: "f32[60]", sum_136: "f32[40]", squeeze_67: "f32[40]", sum_138: "f32[24]", squeeze_64: "f32[24]", sum_140: "f32[20]", squeeze_61: "f32[20]", sum_142: "f32[20]", squeeze_58: "f32[20]", where_38: "f32[32, 72, 1, 1]", where_39: "f32[32, 20, 1, 1]", sum_147: "f32[72]", squeeze_55: "f32[72]", sum_149: "f32[36]", squeeze_52: "f32[36]", sum_151: "f32[36]", squeeze_49: "f32[36]", sum_153: "f32[12]", squeeze_46: "f32[12]", sum_155: "f32[12]", squeeze_43: "f32[12]", sum_157: "f32[36]", squeeze_40: "f32[36]", sum_159: "f32[36]", squeeze_37: "f32[36]", sum_161: "f32[24]", squeeze_34: "f32[24]", sum_163: "f32[16]", squeeze_31: "f32[16]", sum_165: "f32[12]", squeeze_28: "f32[12]", sum_167: "f32[12]", squeeze_25: "f32[12]", sum_169: "f32[48]", squeeze_22: "f32[48]", sum_171: "f32[24]", squeeze_19: "f32[24]", sum_173: "f32[24]", squeeze_16: "f32[24]", sum_175: "f32[8]", squeeze_13: "f32[8]", sum_177: "f32[8]", squeeze_10: "f32[8]", sum_179: "f32[8]", squeeze_7: "f32[8]", sum_181: "f32[8]", squeeze_4: "f32[8]", add_474: "f32[32, 16, 112, 112]", getitem_439: "f32[32, 16, 112, 112]", relu: "f32[32, 16, 112, 112]", full_default: "f32[]", convolution: "f32[32, 16, 112, 112]", unsqueeze_1270: "f32[1, 16, 1, 1]", squeeze_1: "f32[16]", primals_6: "f32[16]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_default: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:832 in forward_head, code: x = self.conv_head(x)
        sum_dim_int_list_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3]);  where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_238);  sum_4 = squeeze_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_1: "f32[80]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_235);  sum_6 = squeeze_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_2: "f32[80]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_232);  sum_8 = squeeze_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_2: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3]);  where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_3: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3]);  where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_3: "f32[480]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_229);  sum_13 = squeeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_4: "f32[480]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_226);  sum_15 = squeeze_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_5: "f32[80]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_223);  sum_17 = squeeze_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_6: "f32[80]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_220);  sum_19 = squeeze_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_7: "f32[480]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_217);  sum_21 = squeeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_8: "f32[480]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_214);  sum_23 = squeeze_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_9: "f32[80]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_211);  sum_25 = squeeze_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_10: "f32[80]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_208);  sum_27 = squeeze_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3]);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_5: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3]);  where_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_11: "f32[480]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_205);  sum_32 = squeeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_12: "f32[480]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_202);  sum_34 = squeeze_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_13: "f32[80]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_199);  sum_36 = squeeze_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_14: "f32[80]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_196);  sum_38 = squeeze_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_15: "f32[480]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_193);  sum_40 = squeeze_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_16: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_190);  sum_42 = squeeze_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        mul_tensor_17: "f32[160]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_187);  sum_44 = squeeze_187 = None
        mul_tensor_18: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_184);  sum_46 = squeeze_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_19: "f32[80]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_181);  sum_48 = squeeze_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_20: "f32[80]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_178);  sum_50 = squeeze_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_6: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3]);  where_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_7: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3]);  where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        mul_tensor_21: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_175);  sum_55 = squeeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_22: "f32[336]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_172);  sum_57 = squeeze_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_23: "f32[336]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_169);  sum_59 = squeeze_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_24: "f32[56]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_166);  sum_61 = squeeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_25: "f32[56]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_163);  sum_63 = squeeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_8: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3]);  where_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_9: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3]);  where_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_26: "f32[336]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_160);  sum_68 = squeeze_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_27: "f32[336]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_157);  sum_70 = squeeze_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        mul_tensor_28: "f32[112]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_154);  sum_72 = squeeze_154 = None
        mul_tensor_29: "f32[80]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_151);  sum_74 = squeeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_30: "f32[56]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_148);  sum_76 = squeeze_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_31: "f32[56]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_145);  sum_78 = squeeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_10: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3]);  where_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_11: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3]);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_32: "f32[240]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_142);  sum_83 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_33: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_139);  sum_85 = squeeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_34: "f32[40]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_136);  sum_87 = squeeze_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_35: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_133);  sum_89 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_36: "f32[92]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_130);  sum_91 = squeeze_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_37: "f32[92]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_127);  sum_93 = squeeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_38: "f32[40]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_124);  sum_95 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_39: "f32[40]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_121);  sum_97 = squeeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_40: "f32[92]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_118);  sum_99 = squeeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_41: "f32[92]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_115);  sum_101 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_42: "f32[40]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_112);  sum_103 = squeeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_43: "f32[40]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_109);  sum_105 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_44: "f32[100]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_106);  sum_107 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_45: "f32[100]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_103);  sum_109 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        mul_tensor_46: "f32[80]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_100);  sum_111 = squeeze_100 = None
        mul_tensor_47: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_97);  sum_113 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_48: "f32[40]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_94);  sum_115 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_49: "f32[40]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_91);  sum_117 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        mul_tensor_50: "f32[240]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_88);  sum_119 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_51: "f32[120]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_85);  sum_121 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_52: "f32[120]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_82);  sum_123 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_53: "f32[20]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_79);  sum_125 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_54: "f32[20]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_76);  sum_127 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_12: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3]);  where_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_13: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3]);  where_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_55: "f32[60]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_73);  sum_132 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_56: "f32[60]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_70);  sum_134 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        mul_tensor_57: "f32[40]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_67);  sum_136 = squeeze_67 = None
        mul_tensor_58: "f32[24]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_64);  sum_138 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_59: "f32[20]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_61);  sum_140 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_60: "f32[20]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_58);  sum_142 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_dim_int_list_14: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3]);  where_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_dim_int_list_15: "f32[20]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3]);  where_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        mul_tensor_61: "f32[72]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_55);  sum_147 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_62: "f32[36]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_52);  sum_149 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_63: "f32[36]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_49);  sum_151 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_64: "f32[12]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_46);  sum_153 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_65: "f32[12]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_43);  sum_155 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_66: "f32[36]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_40);  sum_157 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_67: "f32[36]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_37);  sum_159 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        mul_tensor_68: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_34);  sum_161 = squeeze_34 = None
        mul_tensor_69: "f32[16]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_31);  sum_163 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_70: "f32[12]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_28);  sum_165 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_71: "f32[12]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_25);  sum_167 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        mul_tensor_72: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_22);  sum_169 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_73: "f32[24]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_19);  sum_171 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_74: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_16);  sum_173 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_75: "f32[8]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_13);  sum_175 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_76: "f32[8]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_10);  sum_177 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        mul_tensor_77: "f32[8]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_7);  sum_179 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        mul_tensor_78: "f32[8]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_4);  sum_181 = squeeze_4 = None
        add_tensor: "f32[32, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_474, getitem_439);  add_474 = getitem_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:823 in forward_features, code: x = self.act1(x)
        le_scalar: "b8[32, 16, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[32, 16, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:822 in forward_features, code: x = self.bn1(x)
        sum_dim_int_list_16: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1270);  convolution = unsqueeze_1270 = None
        mul_tensor_79: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_17: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_79, [0, 2, 3]);  mul_tensor_79 = None
        mul_tensor_80: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_16, 2.4912308673469386e-06);  sum_dim_int_list_16 = None
        unsqueeze_default: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_80, 0);  mul_tensor_80 = None
        unsqueeze_default_1: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_81: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_17, 2.4912308673469386e-06)
        mul_tensor_82: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_tensor_83: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_81, mul_tensor_82);  mul_tensor_81 = mul_tensor_82 = None
        unsqueeze_default_3: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_83, 0);  mul_tensor_83 = None
        unsqueeze_default_4: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_84: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_default_6: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_84, 0);  mul_tensor_84 = None
        unsqueeze_default_7: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_85: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_85);  where_self = mul_tensor_85 = None
        sub_tensor_2: "f32[32, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_86: "f32[32, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_87: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_17, squeeze_1);  sum_dim_int_list_17 = squeeze_1 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, mul_tensor, mul_tensor_1, mul_tensor_2, sum_dim_int_list_2, sum_dim_int_list_3, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, sum_dim_int_list_4, sum_dim_int_list_5, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, sum_dim_int_list_6, sum_dim_int_list_7, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, sum_dim_int_list_8, sum_dim_int_list_9, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, sum_dim_int_list_10, sum_dim_int_list_11, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52, mul_tensor_53, mul_tensor_54, sum_dim_int_list_12, sum_dim_int_list_13, mul_tensor_55, mul_tensor_56, mul_tensor_57, mul_tensor_58, mul_tensor_59, mul_tensor_60, sum_dim_int_list_14, sum_dim_int_list_15, mul_tensor_61, mul_tensor_62, mul_tensor_63, mul_tensor_64, mul_tensor_65, mul_tensor_66, mul_tensor_67, mul_tensor_68, mul_tensor_69, mul_tensor_70, mul_tensor_71, mul_tensor_72, mul_tensor_73, mul_tensor_74, mul_tensor_75, mul_tensor_76, mul_tensor_77, mul_tensor_78, mul_tensor_86, mul_tensor_87)


def _default_make_inputs():
    return [
    torch.randn([32, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    torch.randn([32, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([32, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([32, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([32, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 168, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 168, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([100], dtype=torch.float32, device='cuda'),
    torch.randn([100], dtype=torch.float32, device='cuda'),
    torch.randn([100], dtype=torch.float32, device='cuda'),
    torch.randn([100], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([32, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([60], dtype=torch.float32, device='cuda'),
    torch.randn([60], dtype=torch.float32, device='cuda'),
    torch.randn([60], dtype=torch.float32, device='cuda'),
    torch.randn([60], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([32, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 20, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # add_474
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # getitem_439
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # relu
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([32, 16, 112, 112], [200704, 1, 1792, 16]),  # convolution
    torch.randn([1, 16, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
