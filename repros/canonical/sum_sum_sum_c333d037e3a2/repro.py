"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: c333d037e3a2
Shape hash: 4d1cb629
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 1000]", _shape_param_0, permute_140: "f32[4, 7, 7, 768]", mul_126: "f32[4, 7, 7, 768]", view_291: "f32[196, 768]", _shape_param_1, view_294: "f32[196, 3072]", _shape_param_2, view_296: "f32[4, 7, 7, 768]", mul_120: "f32[4, 7, 7, 768]", view_299: "f32[196, 768]", _shape_param_3, fma: "f32[4, 24, 49, 49]", view_310: "f32[196, 2304]", _shape_param_4, _shape_param_5, primals_172: "i64[2401]", view_314: "f32[4, 7, 7, 768]", mul_116: "f32[4, 7, 7, 768]", view_316: "f32[196, 768]", _shape_param_6, view_319: "f32[196, 3072]", _shape_param_7, view_321: "f32[4, 7, 7, 768]", mul_110: "f32[4, 7, 7, 768]", view_324: "f32[196, 768]", _shape_param_8, fma_1: "f32[4, 24, 49, 49]", view_335: "f32[196, 2304]", _shape_param_9, _shape_param_10, primals_158: "i64[2401]", view_339: "f32[4, 7, 7, 768]", mul_106: "f32[4, 7, 7, 768]", view_341: "f32[196, 768]", view_342: "f32[4, 7, 7, 1536]", mul_104: "f32[4, 7, 7, 1536]", view_343: "f32[784, 384]", _shape_param_11, view_346: "f32[784, 1536]", _shape_param_12, view_348: "f32[4, 14, 14, 384]", mul_98: "f32[4, 14, 14, 384]", view_351: "f32[784, 384]", _shape_param_13, fma_2: "f32[16, 12, 49, 49]", view_364: "f32[784, 1152]", _shape_param_14, _shape_param_15, primals_141: "i64[2401]", index_35: "f32[4, 14, 14, 384]", mul_94: "f32[4, 14, 14, 384]", view_370: "f32[784, 384]", _shape_param_16, view_373: "f32[784, 1536]", _shape_param_17, view_375: "f32[4, 14, 14, 384]", mul_88: "f32[4, 14, 14, 384]", view_378: "f32[784, 384]", _shape_param_18, fma_3: "f32[16, 12, 49, 49]", view_389: "f32[784, 1152]", _shape_param_19, _shape_param_20, primals_127: "i64[2401]", view_393: "f32[4, 14, 14, 384]", mul_84: "f32[4, 14, 14, 384]", view_395: "f32[784, 384]", _shape_param_21, view_398: "f32[784, 1536]", _shape_param_22, view_400: "f32[4, 14, 14, 384]", mul_78: "f32[4, 14, 14, 384]", view_403: "f32[784, 384]", _shape_param_23, fma_4: "f32[16, 12, 49, 49]", view_416: "f32[784, 1152]", _shape_param_24, _shape_param_25, primals_113: "i64[2401]", index_39: "f32[4, 14, 14, 384]", mul_74: "f32[4, 14, 14, 384]", view_422: "f32[784, 384]", _shape_param_26, view_425: "f32[784, 1536]", _shape_param_27, view_427: "f32[4, 14, 14, 384]", mul_68: "f32[4, 14, 14, 384]", view_430: "f32[784, 384]", _shape_param_28, fma_5: "f32[16, 12, 49, 49]", view_441: "f32[784, 1152]", _shape_param_29, _shape_param_30, primals_99: "i64[2401]", view_445: "f32[4, 14, 14, 384]", mul_64: "f32[4, 14, 14, 384]", view_447: "f32[784, 384]", _shape_param_31, view_450: "f32[784, 1536]", _shape_param_32, view_452: "f32[4, 14, 14, 384]", mul_58: "f32[4, 14, 14, 384]", view_455: "f32[784, 384]", _shape_param_33, fma_6: "f32[16, 12, 49, 49]", view_468: "f32[784, 1152]", _shape_param_34, _shape_param_35, primals_85: "i64[2401]", index_43: "f32[4, 14, 14, 384]", mul_54: "f32[4, 14, 14, 384]", view_474: "f32[784, 384]", _shape_param_36, view_477: "f32[784, 1536]", _shape_param_37, view_479: "f32[4, 14, 14, 384]", mul_48: "f32[4, 14, 14, 384]", view_482: "f32[784, 384]", _shape_param_38, fma_7: "f32[16, 12, 49, 49]", view_493: "f32[784, 1152]", _shape_param_39, _shape_param_40, primals_71: "i64[2401]", view_497: "f32[4, 14, 14, 384]", mul_44: "f32[4, 14, 14, 384]", view_499: "f32[784, 384]", view_500: "f32[4, 14, 14, 768]", mul_42: "f32[4, 14, 14, 768]", view_501: "f32[3136, 192]", _shape_param_41, view_504: "f32[3136, 768]", _shape_param_42, view_506: "f32[4, 28, 28, 192]", mul_36: "f32[4, 28, 28, 192]", view_509: "f32[3136, 192]", _shape_param_43, fma_8: "f32[64, 6, 49, 49]", view_522: "f32[3136, 576]", _shape_param_44, _shape_param_45, primals_54: "i64[2401]", index_47: "f32[4, 28, 28, 192]", mul_32: "f32[4, 28, 28, 192]", view_528: "f32[3136, 192]", _shape_param_46, view_531: "f32[3136, 768]", _shape_param_47, view_533: "f32[4, 28, 28, 192]", mul_26: "f32[4, 28, 28, 192]", view_536: "f32[3136, 192]", _shape_param_48, fma_9: "f32[64, 6, 49, 49]", view_547: "f32[3136, 576]", _shape_param_49, _shape_param_50, primals_40: "i64[2401]", view_551: "f32[4, 28, 28, 192]", mul_22: "f32[4, 28, 28, 192]", view_553: "f32[3136, 192]", view_554: "f32[4, 28, 28, 384]", mul_20: "f32[4, 28, 28, 384]", view_555: "f32[12544, 96]", _shape_param_51, view_558: "f32[12544, 384]", _shape_param_52, view_560: "f32[4, 56, 56, 96]", mul_14: "f32[4, 56, 56, 96]", view_563: "f32[12544, 96]", _shape_param_53, fma_10: "f32[256, 3, 49, 49]", view_576: "f32[12544, 288]", _shape_param_54, _shape_param_55, primals_23: "i64[2401]", index_51: "f32[4, 56, 56, 96]", mul_10: "f32[4, 56, 56, 96]", view_582: "f32[12544, 96]", _shape_param_56, view_585: "f32[12544, 384]", _shape_param_57, view_587: "f32[4, 56, 56, 96]", mul_5: "f32[4, 56, 56, 96]", view_590: "f32[12544, 96]", _shape_param_58, fma_11: "f32[256, 3, 49, 49]", view_601: "f32[12544, 288]", _shape_param_59, mm_105: "f32[12544, 96]", _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, primals_9: "i64[2401]", primals_6: "f32[96]", convolution: "f32[4, 96, 56, 56]", getitem_1: "f32[4, 56, 56, 1]", rsqrt: "f32[4, 56, 56, 1]", primals_4: "f32[96]", primals_5: "f32[96]", getitem_3: "f32[4, 56, 56, 1]", rsqrt_1: "f32[4, 56, 56, 1]", add_228: "f32[4, 56, 56, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:613 in forward, code: x = self.head(x)
        permute_default: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:609 in forward, code: x = self.norm(x)
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(permute_140, mul_126);  mul_126 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1, 2]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_140, [0, 1, 2]);  permute_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_1: "f32[768, 196]" = torch.ops.aten.permute.default(view_291, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None
        permute_default_2: "f32[3072, 196]" = torch.ops.aten.permute.default(view_294, [1, 0])
        sum_dim_int_list_4: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        reshape_default_2: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(view_296, mul_120);  mul_120 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1, 2]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_296, [0, 1, 2]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_3: "f32[768, 196]" = torch.ops.aten.permute.default(view_299, [1, 0])
        sum_dim_int_list_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_8: "f32[1, 24, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_4: "f32[2304, 196]" = torch.ops.aten.permute.default(view_310, [1, 0])
        sum_dim_int_list_9: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_4: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_4);  sum_dim_int_list_9 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim: "f32[24, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_8, 0);  sum_dim_int_list_8 = None
        permute_default_5: "f32[49, 49, 24]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_5: "f32[2401, 24]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        full_default: "f32[169, 24]" = torch.ops.aten.full.default([169, 24], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[169, 24]" = torch.ops.aten.index_put.default(full_default, [primals_172], reshape_default_5, True);  primals_172 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_2: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(view_314, mul_116);  mul_116 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1, 2]);  mul_tensor_2 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_314, [0, 1, 2]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_6: "f32[768, 196]" = torch.ops.aten.permute.default(view_316, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None
        permute_default_7: "f32[3072, 196]" = torch.ops.aten.permute.default(view_319, [1, 0])
        sum_dim_int_list_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
        reshape_default_7: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None
        mul_tensor_3: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(view_321, mul_110);  mul_110 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1, 2]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_321, [0, 1, 2]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_8: "f32[768, 196]" = torch.ops.aten.permute.default(view_324, [1, 0])
        sum_dim_int_list_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_17: "f32[1, 24, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_1, [0], True);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_9: "f32[2304, 196]" = torch.ops.aten.permute.default(view_335, [1, 0])
        sum_dim_int_list_18: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_9: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_9);  sum_dim_int_list_18 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_1: "f32[24, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_17, 0);  sum_dim_int_list_17 = None
        permute_default_10: "f32[49, 49, 24]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_10: "f32[2401, 24]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_10);  permute_default_10 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_1: "f32[169, 24]" = torch.ops.aten.index_put.default(full_default, [primals_158], reshape_default_10, True);  full_default = primals_158 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_4: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_106);  mul_106 = None
        sum_dim_int_list_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1, 2]);  mul_tensor_4 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1, 2]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        permute_default_11: "f32[768, 196]" = torch.ops.aten.permute.default(view_341, [1, 0]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        mul_tensor_5: "f32[4, 7, 7, 1536]" = torch.ops.aten.mul.Tensor(view_342, mul_104);  mul_104 = None
        sum_dim_int_list_21: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_342, [0, 1, 2]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_12: "f32[384, 784]" = torch.ops.aten.permute.default(view_343, [1, 0])
        sum_dim_int_list_23: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        reshape_default_11: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None
        permute_default_13: "f32[1536, 784]" = torch.ops.aten.permute.default(view_346, [1, 0])
        sum_dim_int_list_24: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
        reshape_default_12: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_12);  sum_dim_int_list_24 = _shape_param_12 = None
        mul_tensor_6: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_348, mul_98);  mul_98 = None
        sum_dim_int_list_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_348, [0, 1, 2]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_14: "f32[384, 784]" = torch.ops.aten.permute.default(view_351, [1, 0])
        sum_dim_int_list_27: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
        reshape_default_13: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_28: "f32[1, 12, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_2, [0], True);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_15: "f32[1152, 784]" = torch.ops.aten.permute.default(view_364, [1, 0])
        sum_dim_int_list_29: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
        reshape_default_14: "f32[1152]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_14);  sum_dim_int_list_29 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_2: "f32[12, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_28, 0);  sum_dim_int_list_28 = None
        permute_default_16: "f32[49, 49, 12]" = torch.ops.aten.permute.default(squeeze_dim_2, [1, 2, 0]);  squeeze_dim_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_15: "f32[2401, 12]" = torch.ops.aten.reshape.default(permute_default_16, _shape_param_15);  permute_default_16 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        full_default_1: "f32[169, 12]" = torch.ops.aten.full.default([169, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[169, 12]" = torch.ops.aten.index_put.default(full_default_1, [primals_141], reshape_default_15, True);  primals_141 = reshape_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_7: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(index_35, mul_94);  mul_94 = None
        sum_dim_int_list_30: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1, 2]);  mul_tensor_7 = None
        sum_dim_int_list_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(index_35, [0, 1, 2]);  index_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_17: "f32[384, 784]" = torch.ops.aten.permute.default(view_370, [1, 0])
        sum_dim_int_list_32: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
        reshape_default_16: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_16);  sum_dim_int_list_32 = _shape_param_16 = None
        permute_default_18: "f32[1536, 784]" = torch.ops.aten.permute.default(view_373, [1, 0])
        sum_dim_int_list_33: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True);  view_373 = None
        reshape_default_17: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_17);  sum_dim_int_list_33 = _shape_param_17 = None
        mul_tensor_8: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_375, mul_88);  mul_88 = None
        sum_dim_int_list_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1, 2]);  mul_tensor_8 = None
        sum_dim_int_list_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_375, [0, 1, 2]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_19: "f32[384, 784]" = torch.ops.aten.permute.default(view_378, [1, 0])
        sum_dim_int_list_36: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
        reshape_default_18: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_18);  sum_dim_int_list_36 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_37: "f32[1, 12, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_3, [0], True);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_20: "f32[1152, 784]" = torch.ops.aten.permute.default(view_389, [1, 0])
        sum_dim_int_list_38: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        reshape_default_19: "f32[1152]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_19);  sum_dim_int_list_38 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_3: "f32[12, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_37, 0);  sum_dim_int_list_37 = None
        permute_default_21: "f32[49, 49, 12]" = torch.ops.aten.permute.default(squeeze_dim_3, [1, 2, 0]);  squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_20: "f32[2401, 12]" = torch.ops.aten.reshape.default(permute_default_21, _shape_param_20);  permute_default_21 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_3: "f32[169, 12]" = torch.ops.aten.index_put.default(full_default_1, [primals_127], reshape_default_20, True);  primals_127 = reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_9: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_393, mul_84);  mul_84 = None
        sum_dim_int_list_39: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1, 2]);  mul_tensor_9 = None
        sum_dim_int_list_40: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_393, [0, 1, 2]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_22: "f32[384, 784]" = torch.ops.aten.permute.default(view_395, [1, 0])
        sum_dim_int_list_41: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        reshape_default_21: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_21);  sum_dim_int_list_41 = _shape_param_21 = None
        permute_default_23: "f32[1536, 784]" = torch.ops.aten.permute.default(view_398, [1, 0])
        sum_dim_int_list_42: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
        reshape_default_22: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_22);  sum_dim_int_list_42 = _shape_param_22 = None
        mul_tensor_10: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_400, mul_78);  mul_78 = None
        sum_dim_int_list_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1, 2]);  mul_tensor_10 = None
        sum_dim_int_list_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_400, [0, 1, 2]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_24: "f32[384, 784]" = torch.ops.aten.permute.default(view_403, [1, 0])
        sum_dim_int_list_45: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
        reshape_default_23: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_23);  sum_dim_int_list_45 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_46: "f32[1, 12, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_4, [0], True);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_25: "f32[1152, 784]" = torch.ops.aten.permute.default(view_416, [1, 0])
        sum_dim_int_list_47: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        reshape_default_24: "f32[1152]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_24);  sum_dim_int_list_47 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_4: "f32[12, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_46, 0);  sum_dim_int_list_46 = None
        permute_default_26: "f32[49, 49, 12]" = torch.ops.aten.permute.default(squeeze_dim_4, [1, 2, 0]);  squeeze_dim_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_25: "f32[2401, 12]" = torch.ops.aten.reshape.default(permute_default_26, _shape_param_25);  permute_default_26 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_4: "f32[169, 12]" = torch.ops.aten.index_put.default(full_default_1, [primals_113], reshape_default_25, True);  primals_113 = reshape_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_11: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(index_39, mul_74);  mul_74 = None
        sum_dim_int_list_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1, 2]);  mul_tensor_11 = None
        sum_dim_int_list_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(index_39, [0, 1, 2]);  index_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_27: "f32[384, 784]" = torch.ops.aten.permute.default(view_422, [1, 0])
        sum_dim_int_list_50: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        reshape_default_26: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_26);  sum_dim_int_list_50 = _shape_param_26 = None
        permute_default_28: "f32[1536, 784]" = torch.ops.aten.permute.default(view_425, [1, 0])
        sum_dim_int_list_51: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        reshape_default_27: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_27);  sum_dim_int_list_51 = _shape_param_27 = None
        mul_tensor_12: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_427, mul_68);  mul_68 = None
        sum_dim_int_list_52: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1, 2]);  mul_tensor_12 = None
        sum_dim_int_list_53: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_29: "f32[384, 784]" = torch.ops.aten.permute.default(view_430, [1, 0])
        sum_dim_int_list_54: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        reshape_default_28: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_28);  sum_dim_int_list_54 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_55: "f32[1, 12, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_5, [0], True);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_30: "f32[1152, 784]" = torch.ops.aten.permute.default(view_441, [1, 0])
        sum_dim_int_list_56: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
        reshape_default_29: "f32[1152]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_29);  sum_dim_int_list_56 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_5: "f32[12, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_55, 0);  sum_dim_int_list_55 = None
        permute_default_31: "f32[49, 49, 12]" = torch.ops.aten.permute.default(squeeze_dim_5, [1, 2, 0]);  squeeze_dim_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_30: "f32[2401, 12]" = torch.ops.aten.reshape.default(permute_default_31, _shape_param_30);  permute_default_31 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_5: "f32[169, 12]" = torch.ops.aten.index_put.default(full_default_1, [primals_99], reshape_default_30, True);  primals_99 = reshape_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_13: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_445, mul_64);  mul_64 = None
        sum_dim_int_list_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1, 2]);  mul_tensor_13 = None
        sum_dim_int_list_58: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_445, [0, 1, 2]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_32: "f32[384, 784]" = torch.ops.aten.permute.default(view_447, [1, 0])
        sum_dim_int_list_59: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        reshape_default_31: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_31);  sum_dim_int_list_59 = _shape_param_31 = None
        permute_default_33: "f32[1536, 784]" = torch.ops.aten.permute.default(view_450, [1, 0])
        sum_dim_int_list_60: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        reshape_default_32: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_32);  sum_dim_int_list_60 = _shape_param_32 = None
        mul_tensor_14: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_452, mul_58);  mul_58 = None
        sum_dim_int_list_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1, 2]);  mul_tensor_14 = None
        sum_dim_int_list_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_452, [0, 1, 2]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_34: "f32[384, 784]" = torch.ops.aten.permute.default(view_455, [1, 0])
        sum_dim_int_list_63: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
        reshape_default_33: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_33);  sum_dim_int_list_63 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_64: "f32[1, 12, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_6, [0], True);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_35: "f32[1152, 784]" = torch.ops.aten.permute.default(view_468, [1, 0])
        sum_dim_int_list_65: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        reshape_default_34: "f32[1152]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_34);  sum_dim_int_list_65 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_6: "f32[12, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_64, 0);  sum_dim_int_list_64 = None
        permute_default_36: "f32[49, 49, 12]" = torch.ops.aten.permute.default(squeeze_dim_6, [1, 2, 0]);  squeeze_dim_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_35: "f32[2401, 12]" = torch.ops.aten.reshape.default(permute_default_36, _shape_param_35);  permute_default_36 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_6: "f32[169, 12]" = torch.ops.aten.index_put.default(full_default_1, [primals_85], reshape_default_35, True);  primals_85 = reshape_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_15: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(index_43, mul_54);  mul_54 = None
        sum_dim_int_list_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1, 2]);  mul_tensor_15 = None
        sum_dim_int_list_67: "f32[384]" = torch.ops.aten.sum.dim_IntList(index_43, [0, 1, 2]);  index_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_37: "f32[384, 784]" = torch.ops.aten.permute.default(view_474, [1, 0])
        sum_dim_int_list_68: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
        reshape_default_36: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_36);  sum_dim_int_list_68 = _shape_param_36 = None
        permute_default_38: "f32[1536, 784]" = torch.ops.aten.permute.default(view_477, [1, 0])
        sum_dim_int_list_69: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
        reshape_default_37: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_37);  sum_dim_int_list_69 = _shape_param_37 = None
        mul_tensor_16: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_479, mul_48);  mul_48 = None
        sum_dim_int_list_70: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1, 2]);  mul_tensor_16 = None
        sum_dim_int_list_71: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_479, [0, 1, 2]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_39: "f32[384, 784]" = torch.ops.aten.permute.default(view_482, [1, 0])
        sum_dim_int_list_72: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_482, [0], True);  view_482 = None
        reshape_default_38: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_38);  sum_dim_int_list_72 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_73: "f32[1, 12, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_7, [0], True);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_40: "f32[1152, 784]" = torch.ops.aten.permute.default(view_493, [1, 0])
        sum_dim_int_list_74: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
        reshape_default_39: "f32[1152]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_39);  sum_dim_int_list_74 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_7: "f32[12, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_73, 0);  sum_dim_int_list_73 = None
        permute_default_41: "f32[49, 49, 12]" = torch.ops.aten.permute.default(squeeze_dim_7, [1, 2, 0]);  squeeze_dim_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_40: "f32[2401, 12]" = torch.ops.aten.reshape.default(permute_default_41, _shape_param_40);  permute_default_41 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_7: "f32[169, 12]" = torch.ops.aten.index_put.default(full_default_1, [primals_71], reshape_default_40, True);  full_default_1 = primals_71 = reshape_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_17: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_497, mul_44);  mul_44 = None
        sum_dim_int_list_75: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1, 2]);  mul_tensor_17 = None
        sum_dim_int_list_76: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_497, [0, 1, 2]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        permute_default_42: "f32[384, 784]" = torch.ops.aten.permute.default(view_499, [1, 0]);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        mul_tensor_18: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(view_500, mul_42);  mul_42 = None
        sum_dim_int_list_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1, 2]);  mul_tensor_18 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_500, [0, 1, 2]);  view_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_43: "f32[192, 3136]" = torch.ops.aten.permute.default(view_501, [1, 0])
        sum_dim_int_list_79: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
        reshape_default_41: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_41);  sum_dim_int_list_79 = _shape_param_41 = None
        permute_default_44: "f32[768, 3136]" = torch.ops.aten.permute.default(view_504, [1, 0])
        sum_dim_int_list_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_42);  sum_dim_int_list_80 = _shape_param_42 = None
        mul_tensor_19: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_506, mul_36);  mul_36 = None
        sum_dim_int_list_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1, 2]);  mul_tensor_19 = None
        sum_dim_int_list_82: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_506, [0, 1, 2]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_45: "f32[192, 3136]" = torch.ops.aten.permute.default(view_509, [1, 0])
        sum_dim_int_list_83: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        reshape_default_43: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_43);  sum_dim_int_list_83 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_84: "f32[1, 6, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_8, [0], True);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_46: "f32[576, 3136]" = torch.ops.aten.permute.default(view_522, [1, 0])
        sum_dim_int_list_85: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
        reshape_default_44: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, _shape_param_44);  sum_dim_int_list_85 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_8: "f32[6, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_84, 0);  sum_dim_int_list_84 = None
        permute_default_47: "f32[49, 49, 6]" = torch.ops.aten.permute.default(squeeze_dim_8, [1, 2, 0]);  squeeze_dim_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_45: "f32[2401, 6]" = torch.ops.aten.reshape.default(permute_default_47, _shape_param_45);  permute_default_47 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        full_default_2: "f32[169, 6]" = torch.ops.aten.full.default([169, 6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_8: "f32[169, 6]" = torch.ops.aten.index_put.default(full_default_2, [primals_54], reshape_default_45, True);  primals_54 = reshape_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_20: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(index_47, mul_32);  mul_32 = None
        sum_dim_int_list_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1, 2]);  mul_tensor_20 = None
        sum_dim_int_list_87: "f32[192]" = torch.ops.aten.sum.dim_IntList(index_47, [0, 1, 2]);  index_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_48: "f32[192, 3136]" = torch.ops.aten.permute.default(view_528, [1, 0])
        sum_dim_int_list_88: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        reshape_default_46: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_46);  sum_dim_int_list_88 = _shape_param_46 = None
        permute_default_49: "f32[768, 3136]" = torch.ops.aten.permute.default(view_531, [1, 0])
        sum_dim_int_list_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_47);  sum_dim_int_list_89 = _shape_param_47 = None
        mul_tensor_21: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_533, mul_26);  mul_26 = None
        sum_dim_int_list_90: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1, 2]);  mul_tensor_21 = None
        sum_dim_int_list_91: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_533, [0, 1, 2]);  view_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_50: "f32[192, 3136]" = torch.ops.aten.permute.default(view_536, [1, 0])
        sum_dim_int_list_92: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
        reshape_default_48: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_48);  sum_dim_int_list_92 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_93: "f32[1, 6, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_9, [0], True);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_51: "f32[576, 3136]" = torch.ops.aten.permute.default(view_547, [1, 0])
        sum_dim_int_list_94: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
        reshape_default_49: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_49);  sum_dim_int_list_94 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_9: "f32[6, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_93, 0);  sum_dim_int_list_93 = None
        permute_default_52: "f32[49, 49, 6]" = torch.ops.aten.permute.default(squeeze_dim_9, [1, 2, 0]);  squeeze_dim_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_50: "f32[2401, 6]" = torch.ops.aten.reshape.default(permute_default_52, _shape_param_50);  permute_default_52 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_9: "f32[169, 6]" = torch.ops.aten.index_put.default(full_default_2, [primals_40], reshape_default_50, True);  full_default_2 = primals_40 = reshape_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_22: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_551, mul_22);  mul_22 = None
        sum_dim_int_list_95: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1, 2]);  mul_tensor_22 = None
        sum_dim_int_list_96: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_551, [0, 1, 2]);  view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        permute_default_53: "f32[192, 3136]" = torch.ops.aten.permute.default(view_553, [1, 0]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        mul_tensor_23: "f32[4, 28, 28, 384]" = torch.ops.aten.mul.Tensor(view_554, mul_20);  mul_20 = None
        sum_dim_int_list_97: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1, 2]);  mul_tensor_23 = None
        sum_dim_int_list_98: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_554, [0, 1, 2]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_54: "f32[96, 12544]" = torch.ops.aten.permute.default(view_555, [1, 0])
        sum_dim_int_list_99: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
        reshape_default_51: "f32[96]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_51);  sum_dim_int_list_99 = _shape_param_51 = None
        permute_default_55: "f32[384, 12544]" = torch.ops.aten.permute.default(view_558, [1, 0])
        sum_dim_int_list_100: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
        reshape_default_52: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_52);  sum_dim_int_list_100 = _shape_param_52 = None
        mul_tensor_24: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(view_560, mul_14);  mul_14 = None
        sum_dim_int_list_101: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1, 2]);  mul_tensor_24 = None
        sum_dim_int_list_102: "f32[96]" = torch.ops.aten.sum.dim_IntList(view_560, [0, 1, 2]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_56: "f32[96, 12544]" = torch.ops.aten.permute.default(view_563, [1, 0])
        sum_dim_int_list_103: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        reshape_default_53: "f32[96]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_53);  sum_dim_int_list_103 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_104: "f32[1, 3, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_10, [0], True);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_57: "f32[288, 12544]" = torch.ops.aten.permute.default(view_576, [1, 0])
        sum_dim_int_list_105: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        reshape_default_54: "f32[288]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, _shape_param_54);  sum_dim_int_list_105 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_10: "f32[3, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_104, 0);  sum_dim_int_list_104 = None
        permute_default_58: "f32[49, 49, 3]" = torch.ops.aten.permute.default(squeeze_dim_10, [1, 2, 0]);  squeeze_dim_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_55: "f32[2401, 3]" = torch.ops.aten.reshape.default(permute_default_58, _shape_param_55);  permute_default_58 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        full_default_3: "f32[169, 3]" = torch.ops.aten.full.default([169, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_10: "f32[169, 3]" = torch.ops.aten.index_put.default(full_default_3, [primals_23], reshape_default_55, True);  primals_23 = reshape_default_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_25: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(index_51, mul_10);  mul_10 = None
        sum_dim_int_list_106: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1, 2]);  mul_tensor_25 = None
        sum_dim_int_list_107: "f32[96]" = torch.ops.aten.sum.dim_IntList(index_51, [0, 1, 2]);  index_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        permute_default_59: "f32[96, 12544]" = torch.ops.aten.permute.default(view_582, [1, 0])
        sum_dim_int_list_108: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
        reshape_default_56: "f32[96]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_56);  sum_dim_int_list_108 = _shape_param_56 = None
        permute_default_60: "f32[384, 12544]" = torch.ops.aten.permute.default(view_585, [1, 0])
        sum_dim_int_list_109: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
        reshape_default_57: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, _shape_param_57);  sum_dim_int_list_109 = _shape_param_57 = None
        mul_tensor_26: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(view_587, mul_5);  mul_5 = None
        sum_dim_int_list_110: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1, 2]);  mul_tensor_26 = None
        sum_dim_int_list_111: "f32[96]" = torch.ops.aten.sum.dim_IntList(view_587, [0, 1, 2]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        permute_default_61: "f32[96, 12544]" = torch.ops.aten.permute.default(view_590, [1, 0])
        sum_dim_int_list_112: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        reshape_default_58: "f32[96]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_58);  sum_dim_int_list_112 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        sum_dim_int_list_113: "f32[1, 3, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_11, [0], True);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        permute_default_62: "f32[288, 12544]" = torch.ops.aten.permute.default(view_601, [1, 0])
        sum_dim_int_list_114: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_601, [0], True);  view_601 = None
        reshape_default_59: "f32[288]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_59);  sum_dim_int_list_114 = _shape_param_59 = None
        reshape_default_60: "f32[256, 49, 96]" = torch.ops.aten.reshape.default(mm_105, _shape_param_60);  mm_105 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        reshape_default_61: "f32[4, 8, 8, 7, 7, 96]" = torch.ops.aten.reshape.default(reshape_default_60, _shape_param_61);  reshape_default_60 = _shape_param_61 = None
        permute_default_63: "f32[4, 8, 7, 8, 7, 96]" = torch.ops.aten.permute.default(reshape_default_61, [0, 1, 3, 2, 4, 5]);  reshape_default_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        clone_default: "f32[4, 8, 7, 8, 7, 96]" = torch.ops.aten.clone.default(permute_default_63, memory_format = torch.contiguous_format);  permute_default_63 = None
        reshape_default_62: "f32[4, 56, 56, 96]" = torch.ops.aten.reshape.default(clone_default, _shape_param_62);  clone_default = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        squeeze_dim_11: "f32[3, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_113, 0);  sum_dim_int_list_113 = None
        permute_default_64: "f32[49, 49, 3]" = torch.ops.aten.permute.default(squeeze_dim_11, [1, 2, 0]);  squeeze_dim_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_63: "f32[2401, 3]" = torch.ops.aten.reshape.default(permute_default_64, _shape_param_63);  permute_default_64 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_put_default_11: "f32[169, 3]" = torch.ops.aten.index_put.default(full_default_3, [primals_9], reshape_default_63, True);  full_default_3 = primals_9 = reshape_default_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor_27: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(reshape_default_62, primals_6);  primals_6 = None
        mul_tensor_28: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 96)
        sum_dim_int_list_115: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_default_65: "f32[4, 56, 56, 96]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:608 in forward, code: x = self.features(x)
        clone_default_1: "f32[4, 56, 56, 96]" = torch.ops.aten.clone.default(permute_default_65, memory_format = torch.contiguous_format)
        sub_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(clone_default_1, getitem_1);  clone_default_1 = None
        mul_tensor_29: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_30: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_29, primals_4);  mul_tensor_29 = None
        add_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(mul_tensor_30, primals_5);  mul_tensor_30 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        sub_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_3);  add_tensor = getitem_3 = None
        mul_tensor_31: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_1);  sub_tensor_1 = None
        mul_tensor_32: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_27, mul_tensor_31);  mul_tensor_27 = None
        sum_dim_int_list_116: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [3], True);  mul_tensor_32 = None
        mul_tensor_33: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_31, sum_dim_int_list_116);  sum_dim_int_list_116 = None
        sub_tensor_2: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(mul_tensor_28, sum_dim_int_list_115);  mul_tensor_28 = sum_dim_int_list_115 = None
        sub_tensor_3: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_33);  sub_tensor_2 = mul_tensor_33 = None
        div_tensor: "f32[4, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 96);  rsqrt_1 = None
        mul_tensor_34: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        mul_tensor_35: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(reshape_default_62, mul_tensor_31);  mul_tensor_31 = None
        sum_dim_int_list_117: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1, 2]);  mul_tensor_35 = None
        sum_dim_int_list_118: "f32[96]" = torch.ops.aten.sum.dim_IntList(reshape_default_62, [0, 1, 2]);  reshape_default_62 = None
        add_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(add_228, mul_tensor_34);  add_228 = mul_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:608 in forward, code: x = self.features(x)
        sub_tensor_4: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(permute_default_65, getitem_1);  permute_default_65 = getitem_1 = None
        mul_tensor_36: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt);  sub_tensor_4 = None
        mul_tensor_37: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None
        mul_tensor_38: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 96)
        sum_dim_int_list_119: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [3], True)
        mul_tensor_39: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_37, mul_tensor_36);  mul_tensor_37 = None
        sum_dim_int_list_120: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [3], True);  mul_tensor_39 = None
        mul_tensor_40: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_36, sum_dim_int_list_120);  sum_dim_int_list_120 = None
        sub_tensor_5: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(mul_tensor_38, sum_dim_int_list_119);  mul_tensor_38 = sum_dim_int_list_119 = None
        sub_tensor_6: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(sub_tensor_5, mul_tensor_40);  sub_tensor_5 = mul_tensor_40 = None
        div_tensor_1: "f32[4, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt, 96);  rsqrt = None
        mul_tensor_41: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_6);  div_tensor_1 = sub_tensor_6 = None
        mul_tensor_42: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_36);  mul_tensor_36 = None
        sum_dim_int_list_121: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1, 2]);  mul_tensor_42 = None
        sum_dim_int_list_122: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_default_66: "f32[4, 96, 56, 56]" = torch.ops.aten.permute.default(mul_tensor_41, [0, 3, 1, 2]);  mul_tensor_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:608 in forward, code: x = self.features(x)
        sum_dim_int_list_123: "f32[96]" = torch.ops.aten.sum.dim_IntList(permute_default_66, [0, 2, 3]);  permute_default_66 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_3, reshape_default_3, permute_default_4, reshape_default_4, index_put_default, sum_dim_int_list_10, sum_dim_int_list_11, permute_default_6, reshape_default_6, permute_default_7, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_8, reshape_default_8, permute_default_9, reshape_default_9, index_put_default_1, sum_dim_int_list_19, sum_dim_int_list_20, permute_default_11, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_12, reshape_default_11, permute_default_13, reshape_default_12, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_14, reshape_default_13, permute_default_15, reshape_default_14, index_put_default_2, sum_dim_int_list_30, sum_dim_int_list_31, permute_default_17, reshape_default_16, permute_default_18, reshape_default_17, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_18, permute_default_20, reshape_default_19, index_put_default_3, sum_dim_int_list_39, sum_dim_int_list_40, permute_default_22, reshape_default_21, permute_default_23, reshape_default_22, sum_dim_int_list_43, sum_dim_int_list_44, permute_default_24, reshape_default_23, permute_default_25, reshape_default_24, index_put_default_4, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_26, permute_default_28, reshape_default_27, sum_dim_int_list_52, sum_dim_int_list_53, permute_default_29, reshape_default_28, permute_default_30, reshape_default_29, index_put_default_5, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_32, reshape_default_31, permute_default_33, reshape_default_32, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_34, reshape_default_33, permute_default_35, reshape_default_34, index_put_default_6, sum_dim_int_list_66, sum_dim_int_list_67, permute_default_37, reshape_default_36, permute_default_38, reshape_default_37, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_39, reshape_default_38, permute_default_40, reshape_default_39, index_put_default_7, sum_dim_int_list_75, sum_dim_int_list_76, permute_default_42, sum_dim_int_list_77, sum_dim_int_list_78, permute_default_43, reshape_default_41, permute_default_44, reshape_default_42, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_45, reshape_default_43, permute_default_46, reshape_default_44, index_put_default_8, sum_dim_int_list_86, sum_dim_int_list_87, permute_default_48, reshape_default_46, permute_default_49, reshape_default_47, sum_dim_int_list_90, sum_dim_int_list_91, permute_default_50, reshape_default_48, permute_default_51, reshape_default_49, index_put_default_9, sum_dim_int_list_95, sum_dim_int_list_96, permute_default_53, sum_dim_int_list_97, sum_dim_int_list_98, permute_default_54, reshape_default_51, permute_default_55, reshape_default_52, sum_dim_int_list_101, sum_dim_int_list_102, permute_default_56, reshape_default_53, permute_default_57, reshape_default_54, index_put_default_10, sum_dim_int_list_106, sum_dim_int_list_107, permute_default_59, reshape_default_56, permute_default_60, reshape_default_57, sum_dim_int_list_110, sum_dim_int_list_111, permute_default_61, reshape_default_58, permute_default_62, reshape_default_59, index_put_default_11, sum_dim_int_list_117, sum_dim_int_list_118, sum_dim_int_list_121, sum_dim_int_list_122, sum_dim_int_list_123)


def _default_make_inputs():
    return [
    torch.randn([4, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    torch.randn(150528, dtype=torch.float32, device='cuda').as_strided([4, 7, 7, 768], [37632, 7, 1, 49]),  # permute_140
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([196, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_2
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_3
    torch.randn([4, 24, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([196, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_4
    [2401, 24],  # _shape_param_5
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([196, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_7
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_8
    torch.randn([4, 24, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([196, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_9
    [2401, 24],  # _shape_param_10
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_11
    torch.randn([784, 1536], dtype=torch.float32, device='cuda'),
    [1536],  # _shape_param_12
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_13
    torch.randn([16, 12, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([784, 1152], dtype=torch.float32, device='cuda'),
    [1152],  # _shape_param_14
    [2401, 12],  # _shape_param_15
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_16
    torch.randn([784, 1536], dtype=torch.float32, device='cuda'),
    [1536],  # _shape_param_17
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_18
    torch.randn([16, 12, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([784, 1152], dtype=torch.float32, device='cuda'),
    [1152],  # _shape_param_19
    [2401, 12],  # _shape_param_20
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_21
    torch.randn([784, 1536], dtype=torch.float32, device='cuda'),
    [1536],  # _shape_param_22
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_23
    torch.randn([16, 12, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([784, 1152], dtype=torch.float32, device='cuda'),
    [1152],  # _shape_param_24
    [2401, 12],  # _shape_param_25
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_26
    torch.randn([784, 1536], dtype=torch.float32, device='cuda'),
    [1536],  # _shape_param_27
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_28
    torch.randn([16, 12, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([784, 1152], dtype=torch.float32, device='cuda'),
    [1152],  # _shape_param_29
    [2401, 12],  # _shape_param_30
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_31
    torch.randn([784, 1536], dtype=torch.float32, device='cuda'),
    [1536],  # _shape_param_32
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_33
    torch.randn([16, 12, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([784, 1152], dtype=torch.float32, device='cuda'),
    [1152],  # _shape_param_34
    [2401, 12],  # _shape_param_35
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_36
    torch.randn([784, 1536], dtype=torch.float32, device='cuda'),
    [1536],  # _shape_param_37
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_38
    torch.randn([16, 12, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([784, 1152], dtype=torch.float32, device='cuda'),
    [1152],  # _shape_param_39
    [2401, 12],  # _shape_param_40
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 192], dtype=torch.float32, device='cuda'),
    [192],  # _shape_param_41
    torch.randn([3136, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_42
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 192], dtype=torch.float32, device='cuda'),
    [192],  # _shape_param_43
    torch.randn([64, 6, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 576], dtype=torch.float32, device='cuda'),
    [576],  # _shape_param_44
    [2401, 6],  # _shape_param_45
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 192], dtype=torch.float32, device='cuda'),
    [192],  # _shape_param_46
    torch.randn([3136, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_47
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 192], dtype=torch.float32, device='cuda'),
    [192],  # _shape_param_48
    torch.randn([64, 6, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 576], dtype=torch.float32, device='cuda'),
    [576],  # _shape_param_49
    [2401, 6],  # _shape_param_50
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([3136, 192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 384], dtype=torch.float32, device='cuda'),
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [96],  # _shape_param_51
    torch.randn([12544, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_52
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [96],  # _shape_param_53
    torch.randn([256, 3, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([12544, 288], dtype=torch.float32, device='cuda'),
    [288],  # _shape_param_54
    [2401, 3],  # _shape_param_55
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [96],  # _shape_param_56
    torch.randn([12544, 384], dtype=torch.float32, device='cuda'),
    [384],  # _shape_param_57
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [96],  # _shape_param_58
    torch.randn([256, 3, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([12544, 288], dtype=torch.float32, device='cuda'),
    [288],  # _shape_param_59
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [256, 49, 96],  # _shape_param_60
    [4, 8, 8, 7, 7, 96],  # _shape_param_61
    [4, 56, 56, 96],  # _shape_param_62
    [2401, 3],  # _shape_param_63
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 96, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
