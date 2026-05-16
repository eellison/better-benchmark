"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_training
Pattern hash: f3fcf5e259f9
Shape hash: 791a6813
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, select_scatter: "f32[32, 1370, 768]", mul_108: "f32[32, 1370, 768]", addmm_47: "f32[43840, 768]", _shape_param_0, mul_115: "f32[32, 1370, 768]", _shape_param_1, view_122: "f32[43840, 768]", _shape_param_2, view_125: "f32[43840, 3072]", _shape_param_3, view_127: "f32[32, 1370, 768]", mul_102: "f32[32, 1370, 768]", addmm_45: "f32[43840, 768]", _shape_param_4, add_89: "f32[32, 1370, 768]", _shape_param_5, view_129: "f32[43840, 768]", getitem_126: "f32[32, 12, 1370, 64]", _shape_param_6, _shape_param_7, _shape_param_8, view_135: "f32[43840, 2304]", _shape_param_9, view_137: "f32[32, 1370, 768]", mul_99: "f32[32, 1370, 768]", addmm_43: "f32[43840, 768]", _shape_param_10, add_90: "f32[32, 1370, 768]", _shape_param_11, view_139: "f32[43840, 768]", _shape_param_12, view_142: "f32[43840, 3072]", _shape_param_13, view_144: "f32[32, 1370, 768]", mul_93: "f32[32, 1370, 768]", addmm_41: "f32[43840, 768]", _shape_param_14, add_93: "f32[32, 1370, 768]", _shape_param_15, view_146: "f32[43840, 768]", getitem_115: "f32[32, 12, 1370, 64]", _shape_param_16, _shape_param_17, _shape_param_18, view_152: "f32[43840, 2304]", _shape_param_19, view_154: "f32[32, 1370, 768]", mul_90: "f32[32, 1370, 768]", addmm_39: "f32[43840, 768]", _shape_param_20, add_94: "f32[32, 1370, 768]", _shape_param_21, view_156: "f32[43840, 768]", _shape_param_22, view_159: "f32[43840, 3072]", _shape_param_23, view_161: "f32[32, 1370, 768]", mul_84: "f32[32, 1370, 768]", addmm_37: "f32[43840, 768]", _shape_param_24, add_97: "f32[32, 1370, 768]", _shape_param_25, view_163: "f32[43840, 768]", getitem_104: "f32[32, 12, 1370, 64]", _shape_param_26, _shape_param_27, _shape_param_28, view_169: "f32[43840, 2304]", _shape_param_29, view_171: "f32[32, 1370, 768]", mul_81: "f32[32, 1370, 768]", addmm_35: "f32[43840, 768]", _shape_param_30, add_98: "f32[32, 1370, 768]", _shape_param_31, view_173: "f32[43840, 768]", _shape_param_32, view_176: "f32[43840, 3072]", _shape_param_33, view_178: "f32[32, 1370, 768]", mul_75: "f32[32, 1370, 768]", addmm_33: "f32[43840, 768]", _shape_param_34, add_101: "f32[32, 1370, 768]", _shape_param_35, view_180: "f32[43840, 768]", getitem_93: "f32[32, 12, 1370, 64]", _shape_param_36, _shape_param_37, _shape_param_38, view_186: "f32[43840, 2304]", _shape_param_39, view_188: "f32[32, 1370, 768]", mul_72: "f32[32, 1370, 768]", addmm_31: "f32[43840, 768]", _shape_param_40, add_102: "f32[32, 1370, 768]", _shape_param_41, view_190: "f32[43840, 768]", _shape_param_42, view_193: "f32[43840, 3072]", _shape_param_43, view_195: "f32[32, 1370, 768]", mul_66: "f32[32, 1370, 768]", addmm_29: "f32[43840, 768]", _shape_param_44, add_105: "f32[32, 1370, 768]", _shape_param_45, view_197: "f32[43840, 768]", getitem_82: "f32[32, 12, 1370, 64]", _shape_param_46, _shape_param_47, _shape_param_48, view_203: "f32[43840, 2304]", _shape_param_49, view_205: "f32[32, 1370, 768]", mul_63: "f32[32, 1370, 768]", addmm_27: "f32[43840, 768]", _shape_param_50, add_106: "f32[32, 1370, 768]", _shape_param_51, view_207: "f32[43840, 768]", _shape_param_52, view_210: "f32[43840, 3072]", _shape_param_53, view_212: "f32[32, 1370, 768]", mul_57: "f32[32, 1370, 768]", addmm_25: "f32[43840, 768]", _shape_param_54, add_109: "f32[32, 1370, 768]", _shape_param_55, view_214: "f32[43840, 768]", getitem_71: "f32[32, 12, 1370, 64]", _shape_param_56, _shape_param_57, _shape_param_58, view_220: "f32[43840, 2304]", _shape_param_59, view_222: "f32[32, 1370, 768]", mul_54: "f32[32, 1370, 768]", addmm_23: "f32[43840, 768]", _shape_param_60, add_110: "f32[32, 1370, 768]", _shape_param_61, view_224: "f32[43840, 768]", _shape_param_62, view_227: "f32[43840, 3072]", _shape_param_63, view_229: "f32[32, 1370, 768]", mul_48: "f32[32, 1370, 768]", addmm_21: "f32[43840, 768]", _shape_param_64, add_113: "f32[32, 1370, 768]", _shape_param_65, view_231: "f32[43840, 768]", getitem_60: "f32[32, 12, 1370, 64]", _shape_param_66, _shape_param_67, _shape_param_68, view_237: "f32[43840, 2304]", _shape_param_69, view_239: "f32[32, 1370, 768]", mul_45: "f32[32, 1370, 768]", addmm_19: "f32[43840, 768]", _shape_param_70, add_114: "f32[32, 1370, 768]", _shape_param_71, view_241: "f32[43840, 768]", _shape_param_72, view_244: "f32[43840, 3072]", _shape_param_73, view_246: "f32[32, 1370, 768]", mul_39: "f32[32, 1370, 768]", addmm_17: "f32[43840, 768]", _shape_param_74, add_117: "f32[32, 1370, 768]", _shape_param_75, view_248: "f32[43840, 768]", getitem_49: "f32[32, 12, 1370, 64]", _shape_param_76, _shape_param_77, _shape_param_78, view_254: "f32[43840, 2304]", _shape_param_79, view_256: "f32[32, 1370, 768]", mul_36: "f32[32, 1370, 768]", addmm_15: "f32[43840, 768]", _shape_param_80, add_118: "f32[32, 1370, 768]", _shape_param_81, view_258: "f32[43840, 768]", _shape_param_82, view_261: "f32[43840, 3072]", _shape_param_83, view_263: "f32[32, 1370, 768]", mul_30: "f32[32, 1370, 768]", addmm_13: "f32[43840, 768]", _shape_param_84, add_121: "f32[32, 1370, 768]", _shape_param_85, view_265: "f32[43840, 768]", getitem_38: "f32[32, 12, 1370, 64]", _shape_param_86, _shape_param_87, _shape_param_88, view_271: "f32[43840, 2304]", _shape_param_89, view_273: "f32[32, 1370, 768]", mul_27: "f32[32, 1370, 768]", addmm_11: "f32[43840, 768]", _shape_param_90, add_122: "f32[32, 1370, 768]", _shape_param_91, view_275: "f32[43840, 768]", _shape_param_92, view_278: "f32[43840, 3072]", _shape_param_93, view_280: "f32[32, 1370, 768]", mul_21: "f32[32, 1370, 768]", addmm_9: "f32[43840, 768]", _shape_param_94, add_125: "f32[32, 1370, 768]", _shape_param_95, view_282: "f32[43840, 768]", getitem_27: "f32[32, 12, 1370, 64]", _shape_param_96, _shape_param_97, _shape_param_98, view_288: "f32[43840, 2304]", _shape_param_99, view_290: "f32[32, 1370, 768]", mul_18: "f32[32, 1370, 768]", addmm_7: "f32[43840, 768]", _shape_param_100, add_126: "f32[32, 1370, 768]", _shape_param_101, view_292: "f32[43840, 768]", _shape_param_102, view_295: "f32[43840, 3072]", _shape_param_103, view_297: "f32[32, 1370, 768]", mul_12: "f32[32, 1370, 768]", addmm_5: "f32[43840, 768]", _shape_param_104, add_129: "f32[32, 1370, 768]", _shape_param_105, view_299: "f32[43840, 768]", getitem_16: "f32[32, 12, 1370, 64]", _shape_param_106, _shape_param_107, _shape_param_108, view_305: "f32[43840, 2304]", _shape_param_109, view_307: "f32[32, 1370, 768]", mul_9: "f32[32, 1370, 768]", addmm_3: "f32[43840, 768]", _shape_param_110, add_130: "f32[32, 1370, 768]", _shape_param_111, view_309: "f32[43840, 768]", _shape_param_112, view_312: "f32[43840, 3072]", _shape_param_113, view_314: "f32[32, 1370, 768]", mul_3: "f32[32, 1370, 768]", addmm_1: "f32[43840, 768]", _shape_param_114, add_133: "f32[32, 1370, 768]", _shape_param_115, view_316: "f32[43840, 768]", getitem_5: "f32[32, 12, 1370, 64]", _shape_param_116, _shape_param_117, _shape_param_118, view_322: "f32[43840, 2304]", _shape_param_119, mm_94: "f32[43840, 768]", _shape_param_120, primals_6: "f32[768]", cat: "f32[32, 1370, 768]", primals_5: "f32[1, 1370, 768]", getitem_1: "f32[32, 1370, 1]", rsqrt: "f32[32, 1370, 1]", _shape_param_121):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(select_scatter, mul_108);  mul_108 = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_1: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_115, reshape_default);  mul_115 = reshape_default = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_1);  sum_dim_int_list_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default: "f32[768, 43840]" = torch.ops.aten.permute.default(view_122, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_2);  sum_dim_int_list_3 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_1: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_125, [1, 0])
        sum_dim_int_list_4: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_2: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_127, mul_102);  mul_102 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_127, [0, 1]);  view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_4: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_4);  addmm_45 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_3: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_89, reshape_default_4);  add_89 = reshape_default_4 = None
        sum_dim_int_list_7: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_5);  sum_dim_int_list_7 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_2: "f32[768, 43840]" = torch.ops.aten.permute.default(view_129, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_3: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        reshape_default_6: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_7: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_7);  reshape_default_6 = _shape_param_7 = None
        sum_dim_int_list_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_4: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_135, [1, 0])
        sum_dim_int_list_9: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        reshape_default_9: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_4: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_137, mul_99);  mul_99 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_10: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_43, _shape_param_10);  addmm_43 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_5: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_90, reshape_default_10);  add_90 = reshape_default_10 = None
        sum_dim_int_list_12: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_11);  sum_dim_int_list_12 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_5: "f32[768, 43840]" = torch.ops.aten.permute.default(view_139, [1, 0])
        sum_dim_int_list_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_139, [0], True);  view_139 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_12);  sum_dim_int_list_13 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_6: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_142, [1, 0])
        sum_dim_int_list_14: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_142, [0], True);  view_142 = None
        reshape_default_13: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_13);  sum_dim_int_list_14 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_6: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_144, mul_93);  mul_93 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_144, [0, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_14: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_41, _shape_param_14);  addmm_41 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_7: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_93, reshape_default_14);  add_93 = reshape_default_14 = None
        sum_dim_int_list_17: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_15);  sum_dim_int_list_17 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_7: "f32[768, 43840]" = torch.ops.aten.permute.default(view_146, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_8: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        reshape_default_16: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_16);  permute_default_8 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_17: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_17);  reshape_default_16 = _shape_param_17 = None
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_18);  sum_dim_int_list_18 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_9: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_152, [1, 0])
        sum_dim_int_list_19: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
        reshape_default_19: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_19);  sum_dim_int_list_19 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_8: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_154, mul_90);  mul_90 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_154, [0, 1]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_20: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_39, _shape_param_20);  addmm_39 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_9: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_94, reshape_default_20);  add_94 = reshape_default_20 = None
        sum_dim_int_list_22: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_21);  sum_dim_int_list_22 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_10: "f32[768, 43840]" = torch.ops.aten.permute.default(view_156, [1, 0])
        sum_dim_int_list_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_22);  sum_dim_int_list_23 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_11: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_159, [1, 0])
        sum_dim_int_list_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
        reshape_default_23: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_23);  sum_dim_int_list_24 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_10: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_161, mul_84);  mul_84 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_161, [0, 1]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_24: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_37, _shape_param_24);  addmm_37 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_11: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_97, reshape_default_24);  add_97 = reshape_default_24 = None
        sum_dim_int_list_27: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_25);  sum_dim_int_list_27 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_12: "f32[768, 43840]" = torch.ops.aten.permute.default(view_163, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_13: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        reshape_default_26: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_13, _shape_param_26);  permute_default_13 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_27: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        sum_dim_int_list_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True);  view_163 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_28);  sum_dim_int_list_28 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_14: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_169, [1, 0])
        sum_dim_int_list_29: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True);  view_169 = None
        reshape_default_29: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_29);  sum_dim_int_list_29 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_12: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_171, mul_81);  mul_81 = None
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_171, [0, 1]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_30: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_30);  addmm_35 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_13: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_98, reshape_default_30);  add_98 = reshape_default_30 = None
        sum_dim_int_list_32: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_31);  sum_dim_int_list_32 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_15: "f32[768, 43840]" = torch.ops.aten.permute.default(view_173, [1, 0])
        sum_dim_int_list_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_32);  sum_dim_int_list_33 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_16: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_176, [1, 0])
        sum_dim_int_list_34: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
        reshape_default_33: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_33);  sum_dim_int_list_34 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_14: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_178, mul_75);  mul_75 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_178, [0, 1]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_34: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_33, _shape_param_34);  addmm_33 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_15: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_101, reshape_default_34);  add_101 = reshape_default_34 = None
        sum_dim_int_list_37: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_35);  sum_dim_int_list_37 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_17: "f32[768, 43840]" = torch.ops.aten.permute.default(view_180, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_18: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        reshape_default_36: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_18, _shape_param_36);  permute_default_18 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_37: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_36, _shape_param_37);  reshape_default_36 = _shape_param_37 = None
        sum_dim_int_list_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_38);  sum_dim_int_list_38 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_19: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_186, [1, 0])
        sum_dim_int_list_39: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        reshape_default_39: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_39);  sum_dim_int_list_39 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_16: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_188, mul_72);  mul_72 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_40: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_31, _shape_param_40);  addmm_31 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_17: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_102, reshape_default_40);  add_102 = reshape_default_40 = None
        sum_dim_int_list_42: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_41);  sum_dim_int_list_42 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_20: "f32[768, 43840]" = torch.ops.aten.permute.default(view_190, [1, 0])
        sum_dim_int_list_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_190, [0], True);  view_190 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_42);  sum_dim_int_list_43 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_21: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_193, [1, 0])
        sum_dim_int_list_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_193, [0], True);  view_193 = None
        reshape_default_43: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_43);  sum_dim_int_list_44 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_18: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_195, mul_66);  mul_66 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 1]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_44: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_29, _shape_param_44);  addmm_29 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_19: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_105, reshape_default_44);  add_105 = reshape_default_44 = None
        sum_dim_int_list_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_45);  sum_dim_int_list_47 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_22: "f32[768, 43840]" = torch.ops.aten.permute.default(view_197, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_23: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        reshape_default_46: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_23, _shape_param_46);  permute_default_23 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_47: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_46, _shape_param_47);  reshape_default_46 = _shape_param_47 = None
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_48);  sum_dim_int_list_48 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_24: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_203, [1, 0])
        sum_dim_int_list_49: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
        reshape_default_49: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_49);  sum_dim_int_list_49 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_20: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_205, mul_63);  mul_63 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_205, [0, 1]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_50: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_27, _shape_param_50);  addmm_27 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_21: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_106, reshape_default_50);  add_106 = reshape_default_50 = None
        sum_dim_int_list_52: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_51: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_51);  sum_dim_int_list_52 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_25: "f32[768, 43840]" = torch.ops.aten.permute.default(view_207, [1, 0])
        sum_dim_int_list_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        reshape_default_52: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_52);  sum_dim_int_list_53 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_26: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_210, [1, 0])
        sum_dim_int_list_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        reshape_default_53: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_53);  sum_dim_int_list_54 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_22: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_212, mul_57);  mul_57 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_54: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_25, _shape_param_54);  addmm_25 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_23: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_109, reshape_default_54);  add_109 = reshape_default_54 = None
        sum_dim_int_list_57: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_55: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_55);  sum_dim_int_list_57 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_27: "f32[768, 43840]" = torch.ops.aten.permute.default(view_214, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_28: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        reshape_default_56: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_28, _shape_param_56);  permute_default_28 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_57: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_56, _shape_param_57);  reshape_default_56 = _shape_param_57 = None
        sum_dim_int_list_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_58);  sum_dim_int_list_58 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_29: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_220, [1, 0])
        sum_dim_int_list_59: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_220, [0], True);  view_220 = None
        reshape_default_59: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_59);  sum_dim_int_list_59 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_24: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_222, mul_54);  mul_54 = None
        sum_dim_int_list_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_222, [0, 1]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_60: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_23, _shape_param_60);  addmm_23 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_25: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_110, reshape_default_60);  add_110 = reshape_default_60 = None
        sum_dim_int_list_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_61: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_61);  sum_dim_int_list_62 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_30: "f32[768, 43840]" = torch.ops.aten.permute.default(view_224, [1, 0])
        sum_dim_int_list_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_62);  sum_dim_int_list_63 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_31: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_227, [1, 0])
        sum_dim_int_list_64: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
        reshape_default_63: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_63);  sum_dim_int_list_64 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_26: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_229, mul_48);  mul_48 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_229, [0, 1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_64: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_21, _shape_param_64);  addmm_21 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_27: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_113, reshape_default_64);  add_113 = reshape_default_64 = None
        sum_dim_int_list_67: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_65: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_65);  sum_dim_int_list_67 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_32: "f32[768, 43840]" = torch.ops.aten.permute.default(view_231, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_33: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        reshape_default_66: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_33, _shape_param_66);  permute_default_33 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_67: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_66, _shape_param_67);  reshape_default_66 = _shape_param_67 = None
        sum_dim_int_list_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        reshape_default_68: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_68);  sum_dim_int_list_68 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_34: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_237, [1, 0])
        sum_dim_int_list_69: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        reshape_default_69: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_69);  sum_dim_int_list_69 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_28: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_239, mul_45);  mul_45 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_239, [0, 1]);  view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_70: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_19, _shape_param_70);  addmm_19 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_29: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_114, reshape_default_70);  add_114 = reshape_default_70 = None
        sum_dim_int_list_72: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_71: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_71);  sum_dim_int_list_72 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_35: "f32[768, 43840]" = torch.ops.aten.permute.default(view_241, [1, 0])
        sum_dim_int_list_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
        reshape_default_72: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_72);  sum_dim_int_list_73 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_36: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_244, [1, 0])
        sum_dim_int_list_74: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
        reshape_default_73: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_73);  sum_dim_int_list_74 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_30: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_246, mul_39);  mul_39 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_246, [0, 1]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_74: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_17, _shape_param_74);  addmm_17 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_31: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_117, reshape_default_74);  add_117 = reshape_default_74 = None
        sum_dim_int_list_77: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_75: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_75);  sum_dim_int_list_77 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_37: "f32[768, 43840]" = torch.ops.aten.permute.default(view_248, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_38: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        reshape_default_76: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_38, _shape_param_76);  permute_default_38 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_77: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_76, _shape_param_77);  reshape_default_76 = _shape_param_77 = None
        sum_dim_int_list_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
        reshape_default_78: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_78);  sum_dim_int_list_78 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_39: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_254, [1, 0])
        sum_dim_int_list_79: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
        reshape_default_79: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_79);  sum_dim_int_list_79 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_32: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_256, mul_36);  mul_36 = None
        sum_dim_int_list_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_256, [0, 1]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_80: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_15, _shape_param_80);  addmm_15 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_33: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_118, reshape_default_80);  add_118 = reshape_default_80 = None
        sum_dim_int_list_82: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_81: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_81);  sum_dim_int_list_82 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_40: "f32[768, 43840]" = torch.ops.aten.permute.default(view_258, [1, 0])
        sum_dim_int_list_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        reshape_default_82: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_82);  sum_dim_int_list_83 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_41: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_261, [1, 0])
        sum_dim_int_list_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
        reshape_default_83: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_83);  sum_dim_int_list_84 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_34: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_263, mul_30);  mul_30 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_84: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_13, _shape_param_84);  addmm_13 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_35: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_121, reshape_default_84);  add_121 = reshape_default_84 = None
        sum_dim_int_list_87: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_85: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_85);  sum_dim_int_list_87 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_42: "f32[768, 43840]" = torch.ops.aten.permute.default(view_265, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_43: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        reshape_default_86: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_43, _shape_param_86);  permute_default_43 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_87: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_86, _shape_param_87);  reshape_default_86 = _shape_param_87 = None
        sum_dim_int_list_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        reshape_default_88: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_88);  sum_dim_int_list_88 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_44: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_271, [1, 0])
        sum_dim_int_list_89: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        reshape_default_89: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_89);  sum_dim_int_list_89 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_36: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_27);  mul_27 = None
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_90: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_11, _shape_param_90);  addmm_11 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_37: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_122, reshape_default_90);  add_122 = reshape_default_90 = None
        sum_dim_int_list_92: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_91: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_91);  sum_dim_int_list_92 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_45: "f32[768, 43840]" = torch.ops.aten.permute.default(view_275, [1, 0])
        sum_dim_int_list_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
        reshape_default_92: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_92);  sum_dim_int_list_93 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_46: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_278, [1, 0])
        sum_dim_int_list_94: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        reshape_default_93: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_93);  sum_dim_int_list_94 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_38: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_280, mul_21);  mul_21 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1]);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_94: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_9, _shape_param_94);  addmm_9 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_39: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_125, reshape_default_94);  add_125 = reshape_default_94 = None
        sum_dim_int_list_97: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_95: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_95);  sum_dim_int_list_97 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_47: "f32[768, 43840]" = torch.ops.aten.permute.default(view_282, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_48: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        reshape_default_96: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_48, _shape_param_96);  permute_default_48 = _shape_param_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_97: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_96, _shape_param_97);  reshape_default_96 = _shape_param_97 = None
        sum_dim_int_list_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_98: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_98);  sum_dim_int_list_98 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_49: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_288, [1, 0])
        sum_dim_int_list_99: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        reshape_default_99: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_99);  sum_dim_int_list_99 = _shape_param_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_40: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_290, mul_18);  mul_18 = None
        sum_dim_int_list_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_290, [0, 1]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_100: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_7, _shape_param_100);  addmm_7 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_41: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_126, reshape_default_100);  add_126 = reshape_default_100 = None
        sum_dim_int_list_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_101: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_101);  sum_dim_int_list_102 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_50: "f32[768, 43840]" = torch.ops.aten.permute.default(view_292, [1, 0])
        sum_dim_int_list_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
        reshape_default_102: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_102);  sum_dim_int_list_103 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_51: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_295, [1, 0])
        sum_dim_int_list_104: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        reshape_default_103: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_103);  sum_dim_int_list_104 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_42: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_297, mul_12);  mul_12 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 1]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_104: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_104);  addmm_5 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_43: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_129, reshape_default_104);  add_129 = reshape_default_104 = None
        sum_dim_int_list_107: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_105: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_105);  sum_dim_int_list_107 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_52: "f32[768, 43840]" = torch.ops.aten.permute.default(view_299, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_53: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        reshape_default_106: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_53, _shape_param_106);  permute_default_53 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_107: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_106, _shape_param_107);  reshape_default_106 = _shape_param_107 = None
        sum_dim_int_list_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        reshape_default_108: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_108);  sum_dim_int_list_108 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_54: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_305, [1, 0])
        sum_dim_int_list_109: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        reshape_default_109: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, _shape_param_109);  sum_dim_int_list_109 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_44: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_307, mul_9);  mul_9 = None
        sum_dim_int_list_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_110: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_110);  addmm_3 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_45: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_130, reshape_default_110);  add_130 = reshape_default_110 = None
        sum_dim_int_list_112: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_111: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_111);  sum_dim_int_list_112 = _shape_param_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_55: "f32[768, 43840]" = torch.ops.aten.permute.default(view_309, [1, 0])
        sum_dim_int_list_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
        reshape_default_112: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_112);  sum_dim_int_list_113 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_56: "f32[3072, 43840]" = torch.ops.aten.permute.default(view_312, [1, 0])
        sum_dim_int_list_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
        reshape_default_113: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_113);  sum_dim_int_list_114 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_46: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(view_314, mul_3);  mul_3 = None
        sum_dim_int_list_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_314, [0, 1]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_114: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_114);  addmm_1 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_47: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(add_133, reshape_default_114);  reshape_default_114 = None
        sum_dim_int_list_117: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_115: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_115);  sum_dim_int_list_117 = _shape_param_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_57: "f32[768, 43840]" = torch.ops.aten.permute.default(view_316, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_58: "f32[32, 1370, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        reshape_default_116: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(permute_default_58, _shape_param_116);  permute_default_58 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_117: "f32[43840, 768]" = torch.ops.aten.reshape.default(reshape_default_116, _shape_param_117);  reshape_default_116 = _shape_param_117 = None
        sum_dim_int_list_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
        reshape_default_118: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_118);  sum_dim_int_list_118 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_59: "f32[2304, 43840]" = torch.ops.aten.permute.default(view_322, [1, 0])
        sum_dim_int_list_119: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
        reshape_default_119: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_119);  sum_dim_int_list_119 = _shape_param_119 = None
        reshape_default_120: "f32[32, 1370, 768]" = torch.ops.aten.reshape.default(mm_94, _shape_param_120);  mm_94 = _shape_param_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_48: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(reshape_default_120, primals_6);  primals_6 = None
        mul_tensor_49: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_48, 768)
        sum_dim_int_list_120: "f32[32, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[32, 1370, 768]" = torch.ops.aten.add.Tensor(cat, primals_5);  cat = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 1370, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_50: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_51: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_48, mul_tensor_50);  mul_tensor_48 = None
        sum_dim_int_list_121: "f32[32, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [2], True);  mul_tensor_51 = None
        mul_tensor_52: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_50, sum_dim_int_list_121);  sum_dim_int_list_121 = None
        sub_tensor_1: "f32[32, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_49, sum_dim_int_list_120);  mul_tensor_49 = sum_dim_int_list_120 = None
        sub_tensor_2: "f32[32, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_52);  sub_tensor_1 = mul_tensor_52 = None
        div_tensor: "f32[32, 1370, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_53: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_54: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(reshape_default_120, mul_tensor_50);  mul_tensor_50 = None
        sum_dim_int_list_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1]);  mul_tensor_54 = None
        sum_dim_int_list_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_120, [0, 1]);  reshape_default_120 = None
        add_tensor_1: "f32[32, 1370, 768]" = torch.ops.aten.add.Tensor(add_133, mul_tensor_53);  add_133 = mul_tensor_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        sum_dim_int_list_124: "f32[1, 1370, 768]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        slice_tensor: "f32[32, 1, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 0, 1)
        slice_tensor_1: "f32[32, 1369, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 1, 1370);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        sum_dim_int_list_125: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default_60: "f32[32, 768, 1369]" = torch.ops.aten.permute.default(slice_tensor_1, [0, 2, 1]);  slice_tensor_1 = None
        reshape_default_121: "f32[32, 768, 37, 37]" = torch.ops.aten.reshape.default(permute_default_60, _shape_param_121);  permute_default_60 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_121, [0, 2, 3]);  reshape_default_121 = None
        return (sum_dim_int_list, sum_dim_int_list_1, reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, sum_dim_int_list_5, sum_dim_int_list_6, reshape_default_5, permute_default_2, reshape_default_7, reshape_default_8, permute_default_4, reshape_default_9, sum_dim_int_list_10, sum_dim_int_list_11, reshape_default_11, permute_default_5, reshape_default_12, permute_default_6, reshape_default_13, sum_dim_int_list_15, sum_dim_int_list_16, reshape_default_15, permute_default_7, reshape_default_17, reshape_default_18, permute_default_9, reshape_default_19, sum_dim_int_list_20, sum_dim_int_list_21, reshape_default_21, permute_default_10, reshape_default_22, permute_default_11, reshape_default_23, sum_dim_int_list_25, sum_dim_int_list_26, reshape_default_25, permute_default_12, reshape_default_27, reshape_default_28, permute_default_14, reshape_default_29, sum_dim_int_list_30, sum_dim_int_list_31, reshape_default_31, permute_default_15, reshape_default_32, permute_default_16, reshape_default_33, sum_dim_int_list_35, sum_dim_int_list_36, reshape_default_35, permute_default_17, reshape_default_37, reshape_default_38, permute_default_19, reshape_default_39, sum_dim_int_list_40, sum_dim_int_list_41, reshape_default_41, permute_default_20, reshape_default_42, permute_default_21, reshape_default_43, sum_dim_int_list_45, sum_dim_int_list_46, reshape_default_45, permute_default_22, reshape_default_47, reshape_default_48, permute_default_24, reshape_default_49, sum_dim_int_list_50, sum_dim_int_list_51, reshape_default_51, permute_default_25, reshape_default_52, permute_default_26, reshape_default_53, sum_dim_int_list_55, sum_dim_int_list_56, reshape_default_55, permute_default_27, reshape_default_57, reshape_default_58, permute_default_29, reshape_default_59, sum_dim_int_list_60, sum_dim_int_list_61, reshape_default_61, permute_default_30, reshape_default_62, permute_default_31, reshape_default_63, sum_dim_int_list_65, sum_dim_int_list_66, reshape_default_65, permute_default_32, reshape_default_67, reshape_default_68, permute_default_34, reshape_default_69, sum_dim_int_list_70, sum_dim_int_list_71, reshape_default_71, permute_default_35, reshape_default_72, permute_default_36, reshape_default_73, sum_dim_int_list_75, sum_dim_int_list_76, reshape_default_75, permute_default_37, reshape_default_77, reshape_default_78, permute_default_39, reshape_default_79, sum_dim_int_list_80, sum_dim_int_list_81, reshape_default_81, permute_default_40, reshape_default_82, permute_default_41, reshape_default_83, sum_dim_int_list_85, sum_dim_int_list_86, reshape_default_85, permute_default_42, reshape_default_87, reshape_default_88, permute_default_44, reshape_default_89, sum_dim_int_list_90, sum_dim_int_list_91, reshape_default_91, permute_default_45, reshape_default_92, permute_default_46, reshape_default_93, sum_dim_int_list_95, sum_dim_int_list_96, reshape_default_95, permute_default_47, reshape_default_97, reshape_default_98, permute_default_49, reshape_default_99, sum_dim_int_list_100, sum_dim_int_list_101, reshape_default_101, permute_default_50, reshape_default_102, permute_default_51, reshape_default_103, sum_dim_int_list_105, sum_dim_int_list_106, reshape_default_105, permute_default_52, reshape_default_107, reshape_default_108, permute_default_54, reshape_default_109, sum_dim_int_list_110, sum_dim_int_list_111, reshape_default_111, permute_default_55, reshape_default_112, permute_default_56, reshape_default_113, sum_dim_int_list_115, sum_dim_int_list_116, reshape_default_115, permute_default_57, reshape_default_117, reshape_default_118, permute_default_59, reshape_default_119, sum_dim_int_list_122, sum_dim_int_list_123, sum_dim_int_list_124, sum_dim_int_list_125, sum_dim_int_list_126)


def _default_make_inputs():
    return [
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_0
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_2
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_3
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_4
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_5
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_126
    [32, 1370, 768],  # _shape_param_6
    [43840, 768],  # _shape_param_7
    [768],  # _shape_param_8
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_9
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_10
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_11
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_12
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_13
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_14
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_15
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_115
    [32, 1370, 768],  # _shape_param_16
    [43840, 768],  # _shape_param_17
    [768],  # _shape_param_18
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_19
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_20
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_21
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_22
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_23
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_24
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_25
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_104
    [32, 1370, 768],  # _shape_param_26
    [43840, 768],  # _shape_param_27
    [768],  # _shape_param_28
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_29
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_30
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_31
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_32
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_33
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_34
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_35
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_93
    [32, 1370, 768],  # _shape_param_36
    [43840, 768],  # _shape_param_37
    [768],  # _shape_param_38
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_39
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_40
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_41
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_42
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_43
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_44
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_45
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_82
    [32, 1370, 768],  # _shape_param_46
    [43840, 768],  # _shape_param_47
    [768],  # _shape_param_48
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_49
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_50
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_51
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_52
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_53
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_54
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_55
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_71
    [32, 1370, 768],  # _shape_param_56
    [43840, 768],  # _shape_param_57
    [768],  # _shape_param_58
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_59
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_60
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_61
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_62
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_63
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_64
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_65
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_60
    [32, 1370, 768],  # _shape_param_66
    [43840, 768],  # _shape_param_67
    [768],  # _shape_param_68
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_69
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_70
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_71
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_72
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_73
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_74
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_75
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_49
    [32, 1370, 768],  # _shape_param_76
    [43840, 768],  # _shape_param_77
    [768],  # _shape_param_78
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_79
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_80
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_81
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_82
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_83
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_84
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_85
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_38
    [32, 1370, 768],  # _shape_param_86
    [43840, 768],  # _shape_param_87
    [768],  # _shape_param_88
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_89
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_90
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_91
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_92
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_93
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_94
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_95
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_27
    [32, 1370, 768],  # _shape_param_96
    [43840, 768],  # _shape_param_97
    [768],  # _shape_param_98
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_99
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_100
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_101
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_102
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_103
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_104
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_105
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_16
    [32, 1370, 768],  # _shape_param_106
    [43840, 768],  # _shape_param_107
    [768],  # _shape_param_108
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_109
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_110
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_111
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_112
    torch.randn([43840, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_113
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_114
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_115
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    torch.randn(33669120, dtype=torch.float32, device='cuda').as_strided([32, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_5
    [32, 1370, 768],  # _shape_param_116
    [43840, 768],  # _shape_param_117
    [768],  # _shape_param_118
    torch.randn([43840, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_119
    torch.randn([43840, 768], dtype=torch.float32, device='cuda'),
    [32, 1370, 768],  # _shape_param_120
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn(44026, dtype=torch.float32, device='cuda').as_strided([32, 1370, 1], [1376, 1, 1]),  # getitem_1
    torch.randn(44026, dtype=torch.float32, device='cuda').as_strided([32, 1370, 1], [1376, 1, 1]),  # rsqrt
    [32, 768, 37, 37],  # _shape_param_121
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
