"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: 64ee492062e1
Shape hash: bce97dde
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1000], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([25344, 3072], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([25344, 768], f32), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([25344, 2304], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([128, 198, 768], f32), T([128, 768, 14, 14], f32, stride=(152064, 1, 10752, 768)), S([1000]), S([1000]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]), S([768]), S([3072]), S([128, 198, 768]), S([25344, 768]), S([768]), S([2304]))"

class Repro(torch.nn.Module):
    def forward(self, div_1: "f32[128, 1000]", add_88: "f32[128, 198, 768]", mul_84: "f32[128, 198, 768]", view_123: "f32[25344, 768]", view_126: "f32[25344, 3072]", view_128: "f32[128, 198, 768]", mul_79: "f32[128, 198, 768]", view_129: "f32[25344, 768]", getitem_126: "f32[128, 12, 198, 64]", view_135: "f32[25344, 2304]", view_137: "f32[128, 198, 768]", mul_77: "f32[128, 198, 768]", view_138: "f32[25344, 768]", view_141: "f32[25344, 3072]", view_143: "f32[128, 198, 768]", mul_72: "f32[128, 198, 768]", view_144: "f32[25344, 768]", getitem_115: "f32[128, 12, 198, 64]", view_150: "f32[25344, 2304]", view_152: "f32[128, 198, 768]", mul_70: "f32[128, 198, 768]", view_153: "f32[25344, 768]", view_156: "f32[25344, 3072]", view_158: "f32[128, 198, 768]", mul_65: "f32[128, 198, 768]", view_159: "f32[25344, 768]", getitem_104: "f32[128, 12, 198, 64]", view_165: "f32[25344, 2304]", view_167: "f32[128, 198, 768]", mul_63: "f32[128, 198, 768]", view_168: "f32[25344, 768]", view_171: "f32[25344, 3072]", view_173: "f32[128, 198, 768]", mul_58: "f32[128, 198, 768]", view_174: "f32[25344, 768]", getitem_93: "f32[128, 12, 198, 64]", view_180: "f32[25344, 2304]", view_182: "f32[128, 198, 768]", mul_56: "f32[128, 198, 768]", view_183: "f32[25344, 768]", view_186: "f32[25344, 3072]", view_188: "f32[128, 198, 768]", mul_51: "f32[128, 198, 768]", view_189: "f32[25344, 768]", getitem_82: "f32[128, 12, 198, 64]", view_195: "f32[25344, 2304]", view_197: "f32[128, 198, 768]", mul_49: "f32[128, 198, 768]", view_198: "f32[25344, 768]", view_201: "f32[25344, 3072]", view_203: "f32[128, 198, 768]", mul_44: "f32[128, 198, 768]", view_204: "f32[25344, 768]", getitem_71: "f32[128, 12, 198, 64]", view_210: "f32[25344, 2304]", view_212: "f32[128, 198, 768]", mul_42: "f32[128, 198, 768]", view_213: "f32[25344, 768]", view_216: "f32[25344, 3072]", view_218: "f32[128, 198, 768]", mul_37: "f32[128, 198, 768]", view_219: "f32[25344, 768]", getitem_60: "f32[128, 12, 198, 64]", view_225: "f32[25344, 2304]", view_227: "f32[128, 198, 768]", mul_35: "f32[128, 198, 768]", view_228: "f32[25344, 768]", view_231: "f32[25344, 3072]", view_233: "f32[128, 198, 768]", mul_30: "f32[128, 198, 768]", view_234: "f32[25344, 768]", getitem_49: "f32[128, 12, 198, 64]", view_240: "f32[25344, 2304]", view_242: "f32[128, 198, 768]", mul_28: "f32[128, 198, 768]", view_243: "f32[25344, 768]", view_246: "f32[25344, 3072]", view_248: "f32[128, 198, 768]", mul_23: "f32[128, 198, 768]", view_249: "f32[25344, 768]", getitem_38: "f32[128, 12, 198, 64]", view_255: "f32[25344, 2304]", view_257: "f32[128, 198, 768]", mul_21: "f32[128, 198, 768]", view_258: "f32[25344, 768]", view_261: "f32[25344, 3072]", view_263: "f32[128, 198, 768]", mul_16: "f32[128, 198, 768]", view_264: "f32[25344, 768]", getitem_27: "f32[128, 12, 198, 64]", view_270: "f32[25344, 2304]", view_272: "f32[128, 198, 768]", mul_14: "f32[128, 198, 768]", view_273: "f32[25344, 768]", view_276: "f32[25344, 3072]", view_278: "f32[128, 198, 768]", mul_9: "f32[128, 198, 768]", view_279: "f32[25344, 768]", getitem_16: "f32[128, 12, 198, 64]", view_285: "f32[25344, 2304]", view_287: "f32[128, 198, 768]", mul_7: "f32[128, 198, 768]", view_288: "f32[25344, 768]", view_291: "f32[25344, 3072]", view_293: "f32[128, 198, 768]", mul_2: "f32[128, 198, 768]", view_294: "f32[25344, 768]", getitem_5: "f32[128, 12, 198, 64]", view_300: "f32[25344, 2304]", view_302: "f32[128, 198, 768]", mul: "f32[128, 198, 768]", add_136: "f32[128, 198, 768]", view_303: "f32[128, 768, 14, 14]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:118 in forward_head, code: x_dist = self.head_dist(x_dist)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(div_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_1, [0], True)
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: x = self.head(x)
        sum_dim_int_list_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_1, [0], True);  div_1 = None
        reshape_default_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add_88, mul_84);  mul_84 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_1: "f32[768, 25344]" = torch.ops.aten.permute.default(view_123, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_2: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_126, [1, 0])
        sum_dim_int_list_5: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_3);  sum_dim_int_list_5 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_128, mul_79);  mul_79 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_3: "f32[768, 25344]" = torch.ops.aten.permute.default(view_129, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_4: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        reshape_default_4: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_5: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        sum_dim_int_list_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_6);  sum_dim_int_list_8 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_5: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_135, [1, 0])
        sum_dim_int_list_9: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        reshape_default_7: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_7);  sum_dim_int_list_9 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_137, mul_77);  mul_77 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_6: "f32[768, 25344]" = torch.ops.aten.permute.default(view_138, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_8);  sum_dim_int_list_12 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_7: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_141, [1, 0])
        sum_dim_int_list_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
        reshape_default_9: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_9);  sum_dim_int_list_13 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_143, mul_72);  mul_72 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_8: "f32[768, 25344]" = torch.ops.aten.permute.default(view_144, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_9: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        reshape_default_10: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_10);  permute_default_9 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_11: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_10, _shape_param_11);  reshape_default_10 = _shape_param_11 = None
        sum_dim_int_list_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_12);  sum_dim_int_list_16 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_10: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_150, [1, 0])
        sum_dim_int_list_17: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
        reshape_default_13: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_13);  sum_dim_int_list_17 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_4: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_70);  mul_70 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_11: "f32[768, 25344]" = torch.ops.aten.permute.default(view_153, [1, 0])
        sum_dim_int_list_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True);  view_153 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_14);  sum_dim_int_list_20 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_12: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_156, [1, 0])
        sum_dim_int_list_21: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        reshape_default_15: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_15);  sum_dim_int_list_21 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_5: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_158, mul_65);  mul_65 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_13: "f32[768, 25344]" = torch.ops.aten.permute.default(view_159, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_14: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        reshape_default_16: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_14, _shape_param_16);  permute_default_14 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_17: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_17);  reshape_default_16 = _shape_param_17 = None
        sum_dim_int_list_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_18);  sum_dim_int_list_24 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_15: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_165, [1, 0])
        sum_dim_int_list_25: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        reshape_default_19: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_19);  sum_dim_int_list_25 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_6: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_167, mul_63);  mul_63 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_167, [0, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_16: "f32[768, 25344]" = torch.ops.aten.permute.default(view_168, [1, 0])
        sum_dim_int_list_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_20);  sum_dim_int_list_28 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_17: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_171, [1, 0])
        sum_dim_int_list_29: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
        reshape_default_21: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_21);  sum_dim_int_list_29 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_7: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_173, mul_58);  mul_58 = None
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_173, [0, 1]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_18: "f32[768, 25344]" = torch.ops.aten.permute.default(view_174, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_19: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        reshape_default_22: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_19, _shape_param_22);  permute_default_19 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_23: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_22, _shape_param_23);  reshape_default_22 = _shape_param_23 = None
        sum_dim_int_list_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_24);  sum_dim_int_list_32 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_20: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_180, [1, 0])
        sum_dim_int_list_33: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        reshape_default_25: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_25);  sum_dim_int_list_33 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_8: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_182, mul_56);  mul_56 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_182, [0, 1]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_21: "f32[768, 25344]" = torch.ops.aten.permute.default(view_183, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_26);  sum_dim_int_list_36 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_22: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_186, [1, 0])
        sum_dim_int_list_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        reshape_default_27: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_27);  sum_dim_int_list_37 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_9: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_188, mul_51);  mul_51 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_23: "f32[768, 25344]" = torch.ops.aten.permute.default(view_189, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_24: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        reshape_default_28: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_24, _shape_param_28);  permute_default_24 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_29: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_28, _shape_param_29);  reshape_default_28 = _shape_param_29 = None
        sum_dim_int_list_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_30);  sum_dim_int_list_40 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_25: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_195, [1, 0])
        sum_dim_int_list_41: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
        reshape_default_31: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_31);  sum_dim_int_list_41 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_10: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_197, mul_49);  mul_49 = None
        sum_dim_int_list_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_197, [0, 1]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_26: "f32[768, 25344]" = torch.ops.aten.permute.default(view_198, [1, 0])
        sum_dim_int_list_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_32);  sum_dim_int_list_44 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_27: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_201, [1, 0])
        sum_dim_int_list_45: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        reshape_default_33: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_33);  sum_dim_int_list_45 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_11: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_203, mul_44);  mul_44 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_203, [0, 1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_28: "f32[768, 25344]" = torch.ops.aten.permute.default(view_204, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_29: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        reshape_default_34: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_29, _shape_param_34);  permute_default_29 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_35: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_34, _shape_param_35);  reshape_default_34 = _shape_param_35 = None
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_36);  sum_dim_int_list_48 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_30: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_210, [1, 0])
        sum_dim_int_list_49: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        reshape_default_37: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_37);  sum_dim_int_list_49 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_12: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_212, mul_42);  mul_42 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_31: "f32[768, 25344]" = torch.ops.aten.permute.default(view_213, [1, 0])
        sum_dim_int_list_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_38);  sum_dim_int_list_52 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_32: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_216, [1, 0])
        sum_dim_int_list_53: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
        reshape_default_39: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_39);  sum_dim_int_list_53 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_13: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_218, mul_37);  mul_37 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_218, [0, 1]);  view_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_33: "f32[768, 25344]" = torch.ops.aten.permute.default(view_219, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_34: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        reshape_default_40: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_34, _shape_param_40);  permute_default_34 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_41: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_40, _shape_param_41);  reshape_default_40 = _shape_param_41 = None
        sum_dim_int_list_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_42);  sum_dim_int_list_56 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_35: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_225, [1, 0])
        sum_dim_int_list_57: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
        reshape_default_43: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_43);  sum_dim_int_list_57 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_14: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_227, mul_35);  mul_35 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_227, [0, 1]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_36: "f32[768, 25344]" = torch.ops.aten.permute.default(view_228, [1, 0])
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
        reshape_default_44: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_44);  sum_dim_int_list_60 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_37: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_231, [1, 0])
        sum_dim_int_list_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        reshape_default_45: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_45);  sum_dim_int_list_61 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_15: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_30);  mul_30 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_38: "f32[768, 25344]" = torch.ops.aten.permute.default(view_234, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_39: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        reshape_default_46: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_39, _shape_param_46);  permute_default_39 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_47: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_46, _shape_param_47);  reshape_default_46 = _shape_param_47 = None
        sum_dim_int_list_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_48);  sum_dim_int_list_64 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_40: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_240, [1, 0])
        sum_dim_int_list_65: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
        reshape_default_49: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_49);  sum_dim_int_list_65 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_16: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_242, mul_28);  mul_28 = None
        sum_dim_int_list_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_242, [0, 1]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_41: "f32[768, 25344]" = torch.ops.aten.permute.default(view_243, [1, 0])
        sum_dim_int_list_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_50);  sum_dim_int_list_68 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_42: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_246, [1, 0])
        sum_dim_int_list_69: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        reshape_default_51: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_51);  sum_dim_int_list_69 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_17: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_248, mul_23);  mul_23 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_248, [0, 1]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_43: "f32[768, 25344]" = torch.ops.aten.permute.default(view_249, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_44: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        reshape_default_52: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_44, _shape_param_52);  permute_default_44 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_53: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_52, _shape_param_53);  reshape_default_52 = _shape_param_53 = None
        sum_dim_int_list_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_54);  sum_dim_int_list_72 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_45: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_255, [1, 0])
        sum_dim_int_list_73: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
        reshape_default_55: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_55);  sum_dim_int_list_73 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_18: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_257, mul_21);  mul_21 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_257, [0, 1]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_46: "f32[768, 25344]" = torch.ops.aten.permute.default(view_258, [1, 0])
        sum_dim_int_list_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_56);  sum_dim_int_list_76 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_47: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_261, [1, 0])
        sum_dim_int_list_77: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
        reshape_default_57: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_57);  sum_dim_int_list_77 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_19: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_263, mul_16);  mul_16 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_48: "f32[768, 25344]" = torch.ops.aten.permute.default(view_264, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_49: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        reshape_default_58: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_49, _shape_param_58);  permute_default_49 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_59: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_58, _shape_param_59);  reshape_default_58 = _shape_param_59 = None
        sum_dim_int_list_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
        reshape_default_60: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_60);  sum_dim_int_list_80 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_50: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_270, [1, 0])
        sum_dim_int_list_81: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
        reshape_default_61: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_61);  sum_dim_int_list_81 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_20: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_272, mul_14);  mul_14 = None
        sum_dim_int_list_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_51: "f32[768, 25344]" = torch.ops.aten.permute.default(view_273, [1, 0])
        sum_dim_int_list_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_62);  sum_dim_int_list_84 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_52: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_276, [1, 0])
        sum_dim_int_list_85: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        reshape_default_63: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, _shape_param_63);  sum_dim_int_list_85 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_21: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_278, mul_9);  mul_9 = None
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_53: "f32[768, 25344]" = torch.ops.aten.permute.default(view_279, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_54: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        reshape_default_64: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_54, _shape_param_64);  permute_default_54 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_65: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_64, _shape_param_65);  reshape_default_64 = _shape_param_65 = None
        sum_dim_int_list_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_66);  sum_dim_int_list_88 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_55: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_285, [1, 0])
        sum_dim_int_list_89: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default_67: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_67);  sum_dim_int_list_89 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_22: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_287, mul_7);  mul_7 = None
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_56: "f32[768, 25344]" = torch.ops.aten.permute.default(view_288, [1, 0])
        sum_dim_int_list_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        reshape_default_68: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_68);  sum_dim_int_list_92 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_57: "f32[3072, 25344]" = torch.ops.aten.permute.default(view_291, [1, 0])
        sum_dim_int_list_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        reshape_default_69: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_69);  sum_dim_int_list_93 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_23: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_293, mul_2);  mul_2 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_293, [0, 1]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_58: "f32[768, 25344]" = torch.ops.aten.permute.default(view_294, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_59: "f32[128, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        reshape_default_70: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(permute_default_59, _shape_param_70);  permute_default_59 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_71: "f32[25344, 768]" = torch.ops.aten.reshape.default(reshape_default_70, _shape_param_71);  reshape_default_70 = _shape_param_71 = None
        sum_dim_int_list_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        reshape_default_72: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_72);  sum_dim_int_list_96 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_60: "f32[2304, 25344]" = torch.ops.aten.permute.default(view_300, [1, 0])
        sum_dim_int_list_97: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        reshape_default_73: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_73);  sum_dim_int_list_97 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_24: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(view_302, mul);  mul = None
        sum_dim_int_list_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_302, [0, 1]);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        sum_dim_int_list_100: "f32[1, 198, 768]" = torch.ops.aten.sum.dim_IntList(add_136, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:105 in _pos_embed, code: x = torch.cat((
        slice_tensor: "f32[128, 1, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 0, 1)
        slice_tensor_1: "f32[128, 1, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 1, 2);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:107 in _pos_embed, code: self.dist_token.expand(x.shape[0], -1, -1),
        sum_dim_int_list_101: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [0], True);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:106 in _pos_embed, code: self.cls_token.expand(x.shape[0], -1, -1),
        sum_dim_int_list_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2, 3]);  view_303 = None
        return (permute_default, reshape_default, reshape_default_1, sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_6, sum_dim_int_list_7, permute_default_3, reshape_default_5, reshape_default_6, permute_default_5, reshape_default_7, sum_dim_int_list_10, sum_dim_int_list_11, permute_default_6, reshape_default_8, permute_default_7, reshape_default_9, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_8, reshape_default_11, reshape_default_12, permute_default_10, reshape_default_13, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_11, reshape_default_14, permute_default_12, reshape_default_15, sum_dim_int_list_22, sum_dim_int_list_23, permute_default_13, reshape_default_17, reshape_default_18, permute_default_15, reshape_default_19, sum_dim_int_list_26, sum_dim_int_list_27, permute_default_16, reshape_default_20, permute_default_17, reshape_default_21, sum_dim_int_list_30, sum_dim_int_list_31, permute_default_18, reshape_default_23, reshape_default_24, permute_default_20, reshape_default_25, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_21, reshape_default_26, permute_default_22, reshape_default_27, sum_dim_int_list_38, sum_dim_int_list_39, permute_default_23, reshape_default_29, reshape_default_30, permute_default_25, reshape_default_31, sum_dim_int_list_42, sum_dim_int_list_43, permute_default_26, reshape_default_32, permute_default_27, reshape_default_33, sum_dim_int_list_46, sum_dim_int_list_47, permute_default_28, reshape_default_35, reshape_default_36, permute_default_30, reshape_default_37, sum_dim_int_list_50, sum_dim_int_list_51, permute_default_31, reshape_default_38, permute_default_32, reshape_default_39, sum_dim_int_list_54, sum_dim_int_list_55, permute_default_33, reshape_default_41, reshape_default_42, permute_default_35, reshape_default_43, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_36, reshape_default_44, permute_default_37, reshape_default_45, sum_dim_int_list_62, sum_dim_int_list_63, permute_default_38, reshape_default_47, reshape_default_48, permute_default_40, reshape_default_49, sum_dim_int_list_66, sum_dim_int_list_67, permute_default_41, reshape_default_50, permute_default_42, reshape_default_51, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_43, reshape_default_53, reshape_default_54, permute_default_45, reshape_default_55, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_46, reshape_default_56, permute_default_47, reshape_default_57, sum_dim_int_list_78, sum_dim_int_list_79, permute_default_48, reshape_default_59, reshape_default_60, permute_default_50, reshape_default_61, sum_dim_int_list_82, sum_dim_int_list_83, permute_default_51, reshape_default_62, permute_default_52, reshape_default_63, sum_dim_int_list_86, sum_dim_int_list_87, permute_default_53, reshape_default_65, reshape_default_66, permute_default_55, reshape_default_67, sum_dim_int_list_90, sum_dim_int_list_91, permute_default_56, reshape_default_68, permute_default_57, reshape_default_69, sum_dim_int_list_94, sum_dim_int_list_95, permute_default_58, reshape_default_71, reshape_default_72, permute_default_60, reshape_default_73, sum_dim_int_list_98, sum_dim_int_list_99, sum_dim_int_list_100, sum_dim_int_list_101, sum_dim_int_list_102, sum_dim_int_list_103)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
