"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_train
Pattern hash: 840e1293a4b6
Shape hash: c6aa41bd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1000], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([25216, 768], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([25216, 192], f32), T([128, 3, 197, 64], f32, stride=(37824, 64, 192, 1)), T([25216, 576], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([128, 197, 192], f32), T([128, 192, 14, 14], f32, stride=(37824, 1, 2688, 192)), S([1000]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]), S([192]), S([768]), S([128, 197, 192]), S([25216, 192]), S([192]), S([576]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", select_scatter: "f32[128, 197, 192]", mul_84: "f32[128, 197, 192]", view_122: "f32[25216, 192]", view_125: "f32[25216, 768]", view_127: "f32[128, 197, 192]", mul_79: "f32[128, 197, 192]", view_128: "f32[25216, 192]", getitem_126: "f32[128, 3, 197, 64]", view_134: "f32[25216, 576]", view_136: "f32[128, 197, 192]", mul_77: "f32[128, 197, 192]", view_137: "f32[25216, 192]", view_140: "f32[25216, 768]", view_142: "f32[128, 197, 192]", mul_72: "f32[128, 197, 192]", view_143: "f32[25216, 192]", getitem_115: "f32[128, 3, 197, 64]", view_149: "f32[25216, 576]", view_151: "f32[128, 197, 192]", mul_70: "f32[128, 197, 192]", view_152: "f32[25216, 192]", view_155: "f32[25216, 768]", view_157: "f32[128, 197, 192]", mul_65: "f32[128, 197, 192]", view_158: "f32[25216, 192]", getitem_104: "f32[128, 3, 197, 64]", view_164: "f32[25216, 576]", view_166: "f32[128, 197, 192]", mul_63: "f32[128, 197, 192]", view_167: "f32[25216, 192]", view_170: "f32[25216, 768]", view_172: "f32[128, 197, 192]", mul_58: "f32[128, 197, 192]", view_173: "f32[25216, 192]", getitem_93: "f32[128, 3, 197, 64]", view_179: "f32[25216, 576]", view_181: "f32[128, 197, 192]", mul_56: "f32[128, 197, 192]", view_182: "f32[25216, 192]", view_185: "f32[25216, 768]", view_187: "f32[128, 197, 192]", mul_51: "f32[128, 197, 192]", view_188: "f32[25216, 192]", getitem_82: "f32[128, 3, 197, 64]", view_194: "f32[25216, 576]", view_196: "f32[128, 197, 192]", mul_49: "f32[128, 197, 192]", view_197: "f32[25216, 192]", view_200: "f32[25216, 768]", view_202: "f32[128, 197, 192]", mul_44: "f32[128, 197, 192]", view_203: "f32[25216, 192]", getitem_71: "f32[128, 3, 197, 64]", view_209: "f32[25216, 576]", view_211: "f32[128, 197, 192]", mul_42: "f32[128, 197, 192]", view_212: "f32[25216, 192]", view_215: "f32[25216, 768]", view_217: "f32[128, 197, 192]", mul_37: "f32[128, 197, 192]", view_218: "f32[25216, 192]", getitem_60: "f32[128, 3, 197, 64]", view_224: "f32[25216, 576]", view_226: "f32[128, 197, 192]", mul_35: "f32[128, 197, 192]", view_227: "f32[25216, 192]", view_230: "f32[25216, 768]", view_232: "f32[128, 197, 192]", mul_30: "f32[128, 197, 192]", view_233: "f32[25216, 192]", getitem_49: "f32[128, 3, 197, 64]", view_239: "f32[25216, 576]", view_241: "f32[128, 197, 192]", mul_28: "f32[128, 197, 192]", view_242: "f32[25216, 192]", view_245: "f32[25216, 768]", view_247: "f32[128, 197, 192]", mul_23: "f32[128, 197, 192]", view_248: "f32[25216, 192]", getitem_38: "f32[128, 3, 197, 64]", view_254: "f32[25216, 576]", view_256: "f32[128, 197, 192]", mul_21: "f32[128, 197, 192]", view_257: "f32[25216, 192]", view_260: "f32[25216, 768]", view_262: "f32[128, 197, 192]", mul_16: "f32[128, 197, 192]", view_263: "f32[25216, 192]", getitem_27: "f32[128, 3, 197, 64]", view_269: "f32[25216, 576]", view_271: "f32[128, 197, 192]", mul_14: "f32[128, 197, 192]", view_272: "f32[25216, 192]", view_275: "f32[25216, 768]", view_277: "f32[128, 197, 192]", mul_9: "f32[128, 197, 192]", view_278: "f32[25216, 192]", getitem_16: "f32[128, 3, 197, 64]", view_284: "f32[25216, 576]", view_286: "f32[128, 197, 192]", mul_7: "f32[128, 197, 192]", view_287: "f32[25216, 192]", view_290: "f32[25216, 768]", view_292: "f32[128, 197, 192]", mul_2: "f32[128, 197, 192]", view_293: "f32[25216, 192]", getitem_5: "f32[128, 3, 197, 64]", view_299: "f32[25216, 576]", view_301: "f32[128, 197, 192]", mul: "f32[128, 197, 192]", add_134: "f32[128, 197, 192]", view_302: "f32[128, 192, 14, 14]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1292 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, mul_84);  mul_84 = None
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_1: "f32[192, 25216]" = torch.ops.aten.permute.default(view_122, [1, 0])
        sum_dim_int_list_3: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
        reshape_default_1: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_2: "f32[768, 25216]" = torch.ops.aten.permute.default(view_125, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_127, mul_79);  mul_79 = None
        sum_dim_int_list_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_127, [0, 1]);  view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_3: "f32[192, 25216]" = torch.ops.aten.permute.default(view_128, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_4: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        reshape_default_3: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_3);  permute_default_4 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_4: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        sum_dim_int_list_7: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True);  view_128 = None
        reshape_default_5: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_5);  sum_dim_int_list_7 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_5: "f32[576, 25216]" = torch.ops.aten.permute.default(view_134, [1, 0])
        sum_dim_int_list_8: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
        reshape_default_6: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_6);  sum_dim_int_list_8 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_136, mul_77);  mul_77 = None
        sum_dim_int_list_9: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_136, [0, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_6: "f32[192, 25216]" = torch.ops.aten.permute.default(view_137, [1, 0])
        sum_dim_int_list_11: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True);  view_137 = None
        reshape_default_7: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_7);  sum_dim_int_list_11 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_7: "f32[768, 25216]" = torch.ops.aten.permute.default(view_140, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_8);  sum_dim_int_list_12 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_142, mul_72);  mul_72 = None
        sum_dim_int_list_13: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_142, [0, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_8: "f32[192, 25216]" = torch.ops.aten.permute.default(view_143, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_9: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        reshape_default_9: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_9);  permute_default_9 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_10: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_9, _shape_param_10);  reshape_default_9 = _shape_param_10 = None
        sum_dim_int_list_15: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True);  view_143 = None
        reshape_default_11: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_11);  sum_dim_int_list_15 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_10: "f32[576, 25216]" = torch.ops.aten.permute.default(view_149, [1, 0])
        sum_dim_int_list_16: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
        reshape_default_12: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_12);  sum_dim_int_list_16 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_151, mul_70);  mul_70 = None
        sum_dim_int_list_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_151, [0, 1]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_11: "f32[192, 25216]" = torch.ops.aten.permute.default(view_152, [1, 0])
        sum_dim_int_list_19: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
        reshape_default_13: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_13);  sum_dim_int_list_19 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_12: "f32[768, 25216]" = torch.ops.aten.permute.default(view_155, [1, 0])
        sum_dim_int_list_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_155, [0], True);  view_155 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_14);  sum_dim_int_list_20 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_157, mul_65);  mul_65 = None
        sum_dim_int_list_21: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_157, [0, 1]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_13: "f32[192, 25216]" = torch.ops.aten.permute.default(view_158, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_14: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        reshape_default_15: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_14, _shape_param_15);  permute_default_14 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_16: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_15, _shape_param_16);  reshape_default_15 = _shape_param_16 = None
        sum_dim_int_list_23: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True);  view_158 = None
        reshape_default_17: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_17);  sum_dim_int_list_23 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_15: "f32[576, 25216]" = torch.ops.aten.permute.default(view_164, [1, 0])
        sum_dim_int_list_24: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True);  view_164 = None
        reshape_default_18: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_18);  sum_dim_int_list_24 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_6: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_166, mul_63);  mul_63 = None
        sum_dim_int_list_25: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_166, [0, 1]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_16: "f32[192, 25216]" = torch.ops.aten.permute.default(view_167, [1, 0])
        sum_dim_int_list_27: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        reshape_default_19: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_19);  sum_dim_int_list_27 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_17: "f32[768, 25216]" = torch.ops.aten.permute.default(view_170, [1, 0])
        sum_dim_int_list_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_20);  sum_dim_int_list_28 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_7: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_172, mul_58);  mul_58 = None
        sum_dim_int_list_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_172, [0, 1]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_18: "f32[192, 25216]" = torch.ops.aten.permute.default(view_173, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_19: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        reshape_default_21: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_19, _shape_param_21);  permute_default_19 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_22: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_21, _shape_param_22);  reshape_default_21 = _shape_param_22 = None
        sum_dim_int_list_31: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        reshape_default_23: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_23);  sum_dim_int_list_31 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_20: "f32[576, 25216]" = torch.ops.aten.permute.default(view_179, [1, 0])
        sum_dim_int_list_32: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
        reshape_default_24: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_24);  sum_dim_int_list_32 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_8: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_181, mul_56);  mul_56 = None
        sum_dim_int_list_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_181, [0, 1]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_21: "f32[192, 25216]" = torch.ops.aten.permute.default(view_182, [1, 0])
        sum_dim_int_list_35: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
        reshape_default_25: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_25);  sum_dim_int_list_35 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_22: "f32[768, 25216]" = torch.ops.aten.permute.default(view_185, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_26);  sum_dim_int_list_36 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_9: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_187, mul_51);  mul_51 = None
        sum_dim_int_list_37: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_187, [0, 1]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_23: "f32[192, 25216]" = torch.ops.aten.permute.default(view_188, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_24: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        reshape_default_27: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_24, _shape_param_27);  permute_default_24 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_28: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_27, _shape_param_28);  reshape_default_27 = _shape_param_28 = None
        sum_dim_int_list_39: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        reshape_default_29: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_29);  sum_dim_int_list_39 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_25: "f32[576, 25216]" = torch.ops.aten.permute.default(view_194, [1, 0])
        sum_dim_int_list_40: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
        reshape_default_30: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_30);  sum_dim_int_list_40 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_10: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_196, mul_49);  mul_49 = None
        sum_dim_int_list_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_196, [0, 1]);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_26: "f32[192, 25216]" = torch.ops.aten.permute.default(view_197, [1, 0])
        sum_dim_int_list_43: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
        reshape_default_31: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_31);  sum_dim_int_list_43 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_27: "f32[768, 25216]" = torch.ops.aten.permute.default(view_200, [1, 0])
        sum_dim_int_list_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_32);  sum_dim_int_list_44 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_11: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_202, mul_44);  mul_44 = None
        sum_dim_int_list_45: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_202, [0, 1]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_28: "f32[192, 25216]" = torch.ops.aten.permute.default(view_203, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_29: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        reshape_default_33: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_29, _shape_param_33);  permute_default_29 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_34: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_33, _shape_param_34);  reshape_default_33 = _shape_param_34 = None
        sum_dim_int_list_47: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
        reshape_default_35: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_35);  sum_dim_int_list_47 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_30: "f32[576, 25216]" = torch.ops.aten.permute.default(view_209, [1, 0])
        sum_dim_int_list_48: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
        reshape_default_36: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_36);  sum_dim_int_list_48 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_12: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_211, mul_42);  mul_42 = None
        sum_dim_int_list_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_31: "f32[192, 25216]" = torch.ops.aten.permute.default(view_212, [1, 0])
        sum_dim_int_list_51: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
        reshape_default_37: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_37);  sum_dim_int_list_51 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_32: "f32[768, 25216]" = torch.ops.aten.permute.default(view_215, [1, 0])
        sum_dim_int_list_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_38);  sum_dim_int_list_52 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_13: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_217, mul_37);  mul_37 = None
        sum_dim_int_list_53: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_33: "f32[192, 25216]" = torch.ops.aten.permute.default(view_218, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_34: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        reshape_default_39: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_34, _shape_param_39);  permute_default_34 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_40: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_39, _shape_param_40);  reshape_default_39 = _shape_param_40 = None
        sum_dim_int_list_55: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        reshape_default_41: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_41);  sum_dim_int_list_55 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_35: "f32[576, 25216]" = torch.ops.aten.permute.default(view_224, [1, 0])
        sum_dim_int_list_56: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
        reshape_default_42: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_42);  sum_dim_int_list_56 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_14: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_226, mul_35);  mul_35 = None
        sum_dim_int_list_57: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_36: "f32[192, 25216]" = torch.ops.aten.permute.default(view_227, [1, 0])
        sum_dim_int_list_59: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
        reshape_default_43: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_43);  sum_dim_int_list_59 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_37: "f32[768, 25216]" = torch.ops.aten.permute.default(view_230, [1, 0])
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
        reshape_default_44: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_44);  sum_dim_int_list_60 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_15: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_232, mul_30);  mul_30 = None
        sum_dim_int_list_61: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_232, [0, 1]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_38: "f32[192, 25216]" = torch.ops.aten.permute.default(view_233, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_39: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        reshape_default_45: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_39, _shape_param_45);  permute_default_39 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_46: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_45, _shape_param_46);  reshape_default_45 = _shape_param_46 = None
        sum_dim_int_list_63: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
        reshape_default_47: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_47);  sum_dim_int_list_63 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_40: "f32[576, 25216]" = torch.ops.aten.permute.default(view_239, [1, 0])
        sum_dim_int_list_64: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
        reshape_default_48: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_48);  sum_dim_int_list_64 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_16: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_241, mul_28);  mul_28 = None
        sum_dim_int_list_65: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_41: "f32[192, 25216]" = torch.ops.aten.permute.default(view_242, [1, 0])
        sum_dim_int_list_67: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
        reshape_default_49: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_49);  sum_dim_int_list_67 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_42: "f32[768, 25216]" = torch.ops.aten.permute.default(view_245, [1, 0])
        sum_dim_int_list_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_50);  sum_dim_int_list_68 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_17: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_247, mul_23);  mul_23 = None
        sum_dim_int_list_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_247, [0, 1]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_43: "f32[192, 25216]" = torch.ops.aten.permute.default(view_248, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_44: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        reshape_default_51: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_44, _shape_param_51);  permute_default_44 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_52: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_51, _shape_param_52);  reshape_default_51 = _shape_param_52 = None
        sum_dim_int_list_71: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
        reshape_default_53: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_53);  sum_dim_int_list_71 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_45: "f32[576, 25216]" = torch.ops.aten.permute.default(view_254, [1, 0])
        sum_dim_int_list_72: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
        reshape_default_54: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_54);  sum_dim_int_list_72 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_18: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_256, mul_21);  mul_21 = None
        sum_dim_int_list_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_256, [0, 1]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_46: "f32[192, 25216]" = torch.ops.aten.permute.default(view_257, [1, 0])
        sum_dim_int_list_75: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
        reshape_default_55: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_55);  sum_dim_int_list_75 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_47: "f32[768, 25216]" = torch.ops.aten.permute.default(view_260, [1, 0])
        sum_dim_int_list_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_56);  sum_dim_int_list_76 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_19: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_262, mul_16);  mul_16 = None
        sum_dim_int_list_77: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_262, [0, 1]);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_48: "f32[192, 25216]" = torch.ops.aten.permute.default(view_263, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_49: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        reshape_default_57: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_49, _shape_param_57);  permute_default_49 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_58: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_57, _shape_param_58);  reshape_default_57 = _shape_param_58 = None
        sum_dim_int_list_79: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
        reshape_default_59: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_59);  sum_dim_int_list_79 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_50: "f32[576, 25216]" = torch.ops.aten.permute.default(view_269, [1, 0])
        sum_dim_int_list_80: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
        reshape_default_60: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_60);  sum_dim_int_list_80 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_20: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_271, mul_14);  mul_14 = None
        sum_dim_int_list_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_51: "f32[192, 25216]" = torch.ops.aten.permute.default(view_272, [1, 0])
        sum_dim_int_list_83: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
        reshape_default_61: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_61);  sum_dim_int_list_83 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_52: "f32[768, 25216]" = torch.ops.aten.permute.default(view_275, [1, 0])
        sum_dim_int_list_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_62);  sum_dim_int_list_84 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_21: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_277, mul_9);  mul_9 = None
        sum_dim_int_list_85: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_277, [0, 1]);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_53: "f32[192, 25216]" = torch.ops.aten.permute.default(view_278, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_54: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        reshape_default_63: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_54, _shape_param_63);  permute_default_54 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_64: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_63, _shape_param_64);  reshape_default_63 = _shape_param_64 = None
        sum_dim_int_list_87: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        reshape_default_65: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_65);  sum_dim_int_list_87 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_55: "f32[576, 25216]" = torch.ops.aten.permute.default(view_284, [1, 0])
        sum_dim_int_list_88: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
        reshape_default_66: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_66);  sum_dim_int_list_88 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_22: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_286, mul_7);  mul_7 = None
        sum_dim_int_list_89: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_286, [0, 1]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_56: "f32[192, 25216]" = torch.ops.aten.permute.default(view_287, [1, 0])
        sum_dim_int_list_91: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
        reshape_default_67: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_67);  sum_dim_int_list_91 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_57: "f32[768, 25216]" = torch.ops.aten.permute.default(view_290, [1, 0])
        sum_dim_int_list_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
        reshape_default_68: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_68);  sum_dim_int_list_92 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_23: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_292, mul_2);  mul_2 = None
        sum_dim_int_list_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_292, [0, 1]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_58: "f32[192, 25216]" = torch.ops.aten.permute.default(view_293, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_59: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        reshape_default_69: "f32[128, 197, 192]" = torch.ops.aten.reshape.default(permute_default_59, _shape_param_69);  permute_default_59 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_70: "f32[25216, 192]" = torch.ops.aten.reshape.default(reshape_default_69, _shape_param_70);  reshape_default_69 = _shape_param_70 = None
        sum_dim_int_list_95: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        reshape_default_71: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_71);  sum_dim_int_list_95 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_60: "f32[576, 25216]" = torch.ops.aten.permute.default(view_299, [1, 0])
        sum_dim_int_list_96: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        reshape_default_72: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_72);  sum_dim_int_list_96 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_24: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_301, mul);  mul = None
        sum_dim_int_list_97: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        sum_dim_int_list_99: "f32[1, 197, 192]" = torch.ops.aten.sum.dim_IntList(add_134, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        slice_tensor: "f32[128, 1, 192]" = torch.ops.aten.slice.Tensor(add_134, 1, 0, 1);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        sum_dim_int_list_100: "f32[1, 1, 192]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_101: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_302, [0, 2, 3]);  view_302 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_3, reshape_default_4, reshape_default_5, permute_default_5, reshape_default_6, sum_dim_int_list_9, sum_dim_int_list_10, permute_default_6, reshape_default_7, permute_default_7, reshape_default_8, sum_dim_int_list_13, sum_dim_int_list_14, permute_default_8, reshape_default_10, reshape_default_11, permute_default_10, reshape_default_12, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_11, reshape_default_13, permute_default_12, reshape_default_14, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_13, reshape_default_16, reshape_default_17, permute_default_15, reshape_default_18, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_16, reshape_default_19, permute_default_17, reshape_default_20, sum_dim_int_list_29, sum_dim_int_list_30, permute_default_18, reshape_default_22, reshape_default_23, permute_default_20, reshape_default_24, sum_dim_int_list_33, sum_dim_int_list_34, permute_default_21, reshape_default_25, permute_default_22, reshape_default_26, sum_dim_int_list_37, sum_dim_int_list_38, permute_default_23, reshape_default_28, reshape_default_29, permute_default_25, reshape_default_30, sum_dim_int_list_41, sum_dim_int_list_42, permute_default_26, reshape_default_31, permute_default_27, reshape_default_32, sum_dim_int_list_45, sum_dim_int_list_46, permute_default_28, reshape_default_34, reshape_default_35, permute_default_30, reshape_default_36, sum_dim_int_list_49, sum_dim_int_list_50, permute_default_31, reshape_default_37, permute_default_32, reshape_default_38, sum_dim_int_list_53, sum_dim_int_list_54, permute_default_33, reshape_default_40, reshape_default_41, permute_default_35, reshape_default_42, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_36, reshape_default_43, permute_default_37, reshape_default_44, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_38, reshape_default_46, reshape_default_47, permute_default_40, reshape_default_48, sum_dim_int_list_65, sum_dim_int_list_66, permute_default_41, reshape_default_49, permute_default_42, reshape_default_50, sum_dim_int_list_69, sum_dim_int_list_70, permute_default_43, reshape_default_52, reshape_default_53, permute_default_45, reshape_default_54, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_46, reshape_default_55, permute_default_47, reshape_default_56, sum_dim_int_list_77, sum_dim_int_list_78, permute_default_48, reshape_default_58, reshape_default_59, permute_default_50, reshape_default_60, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_51, reshape_default_61, permute_default_52, reshape_default_62, sum_dim_int_list_85, sum_dim_int_list_86, permute_default_53, reshape_default_64, reshape_default_65, permute_default_55, reshape_default_66, sum_dim_int_list_89, sum_dim_int_list_90, permute_default_56, reshape_default_67, permute_default_57, reshape_default_68, sum_dim_int_list_93, sum_dim_int_list_94, permute_default_58, reshape_default_70, reshape_default_71, permute_default_60, reshape_default_72, sum_dim_int_list_97, sum_dim_int_list_98, sum_dim_int_list_99, sum_dim_int_list_100, sum_dim_int_list_101)


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
