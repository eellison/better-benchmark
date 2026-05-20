"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph5
Pattern hash: c128521bb4f8
Shape hash: bd3c5e5a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 32000], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 3072], f32), T([1, 512, 768], f32), T([1, 512, 768], f32), T([512, 768], f32), T([512, 768], f32), T([768], f32), T([1, 512, 768], f32), T([1, 512, 1], f32), T([1, 512], i64, max=512), T([], f32), T([1, 512], i64, max=4), T([1, 512], i64, max=32000), T([32000, 768], f32), S([32000]), S([768]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([3072]), S([768]), S([1, 512, 768]), S([1, 512]))"

class Repro(torch.nn.Module):
    def forward(self, sub_2: "f32[512, 32000]", view_3: "f32[1, 512, 768]", mul_8: "f32[1, 512, 768]", view_5: "f32[512, 768]", view_7: "f32[1, 512, 768]", arg131_1: "f32[1, 512, 768]", view_8: "f32[512, 768]", view_12: "f32[512, 3072]", add_8: "f32[1, 512, 768]", arg126_1: "f32[1, 512, 768]", add_9: "f32[1, 512, 768]", arg125_1: "f32[1, 512, 768]", view_15: "f32[512, 768]", view_19: "f32[512, 3072]", add_14: "f32[1, 512, 768]", arg120_1: "f32[1, 512, 768]", add_15: "f32[1, 512, 768]", arg119_1: "f32[1, 512, 768]", view_22: "f32[512, 768]", view_26: "f32[512, 3072]", add_20: "f32[1, 512, 768]", arg114_1: "f32[1, 512, 768]", add_21: "f32[1, 512, 768]", arg113_1: "f32[1, 512, 768]", view_29: "f32[512, 768]", view_33: "f32[512, 3072]", add_26: "f32[1, 512, 768]", arg108_1: "f32[1, 512, 768]", add_27: "f32[1, 512, 768]", arg107_1: "f32[1, 512, 768]", view_36: "f32[512, 768]", view_40: "f32[512, 3072]", add_32: "f32[1, 512, 768]", arg102_1: "f32[1, 512, 768]", add_33: "f32[1, 512, 768]", arg101_1: "f32[1, 512, 768]", view_43: "f32[512, 768]", view_47: "f32[512, 3072]", add_38: "f32[1, 512, 768]", arg96_1: "f32[1, 512, 768]", add_39: "f32[1, 512, 768]", arg95_1: "f32[1, 512, 768]", view_50: "f32[512, 768]", view_54: "f32[512, 3072]", add_44: "f32[1, 512, 768]", arg90_1: "f32[1, 512, 768]", add_45: "f32[1, 512, 768]", arg89_1: "f32[1, 512, 768]", view_57: "f32[512, 768]", view_61: "f32[512, 3072]", add_50: "f32[1, 512, 768]", arg84_1: "f32[1, 512, 768]", add_51: "f32[1, 512, 768]", arg83_1: "f32[1, 512, 768]", view_64: "f32[512, 768]", view_68: "f32[512, 3072]", add_56: "f32[1, 512, 768]", arg78_1: "f32[1, 512, 768]", add_57: "f32[1, 512, 768]", arg77_1: "f32[1, 512, 768]", view_71: "f32[512, 768]", view_75: "f32[512, 3072]", add_62: "f32[1, 512, 768]", arg72_1: "f32[1, 512, 768]", add_63: "f32[1, 512, 768]", arg71_1: "f32[1, 512, 768]", view_78: "f32[512, 768]", view_82: "f32[512, 3072]", add_68: "f32[1, 512, 768]", arg66_1: "f32[1, 512, 768]", add_69: "f32[1, 512, 768]", arg65_1: "f32[1, 512, 768]", view_85: "f32[512, 768]", view_89: "f32[512, 3072]", add_74: "f32[1, 512, 768]", arg60_1: "f32[1, 512, 768]", view_92: "f32[512, 768]", mm_52: "f32[512, 768]", arg4_1: "f32[768]", arg57_1: "f32[1, 512, 768]", arg165_1: "f32[1, 512, 1]", arg2_1: "i64[1, 512]", full_1: "f32[]", arg1_1: "i64[1, 512]", arg0_1: "i64[1, 512]", mm_1: "f32[32000, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(sub_2, [0], True);  sub_2 = None
        view_default: "f32[32000]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_3, mul_8);  mul_8 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_3, [0, 1]);  view_3 = None
        permute_default: "f32[768, 512]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True);  view_5 = None
        view_default_1: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_7, arg131_1);  arg131_1 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_7, [0, 1]);  view_7 = None
        permute_default_1: "f32[768, 512]" = torch.ops.aten.permute.default(view_8, [1, 0])
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True);  view_8 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None
        permute_default_2: "f32[3072, 512]" = torch.ops.aten.permute.default(view_12, [1, 0])
        sum_dim_int_list_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True);  view_12 = None
        view_default_3: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_8, arg126_1);  arg126_1 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1]);  add_8 = None
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_9, arg125_1);  arg125_1 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        permute_default_3: "f32[768, 512]" = torch.ops.aten.permute.default(view_15, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True);  view_15 = None
        view_default_4: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_12, _shape_param_4);  sum_dim_int_list_12 = _shape_param_4 = None
        permute_default_4: "f32[3072, 512]" = torch.ops.aten.permute.default(view_19, [1, 0])
        sum_dim_int_list_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        view_default_5: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_13, _shape_param_5);  sum_dim_int_list_13 = _shape_param_5 = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_14, arg120_1);  arg120_1 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        mul_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_15, arg119_1);  arg119_1 = None
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_15, [0, 1]);  add_15 = None
        permute_default_5: "f32[768, 512]" = torch.ops.aten.permute.default(view_22, [1, 0])
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_default_6: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_18, _shape_param_6);  sum_dim_int_list_18 = _shape_param_6 = None
        permute_default_6: "f32[3072, 512]" = torch.ops.aten.permute.default(view_26, [1, 0])
        sum_dim_int_list_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True);  view_26 = None
        view_default_7: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_19, _shape_param_7);  sum_dim_int_list_19 = _shape_param_7 = None
        mul_tensor_6: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_20, arg114_1);  arg114_1 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_20, [0, 1]);  add_20 = None
        mul_tensor_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_21, arg113_1);  arg113_1 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_21, [0, 1]);  add_21 = None
        permute_default_7: "f32[768, 512]" = torch.ops.aten.permute.default(view_29, [1, 0])
        sum_dim_int_list_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_29, [0], True);  view_29 = None
        view_default_8: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_24, _shape_param_8);  sum_dim_int_list_24 = _shape_param_8 = None
        permute_default_8: "f32[3072, 512]" = torch.ops.aten.permute.default(view_33, [1, 0])
        sum_dim_int_list_25: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_33, [0], True);  view_33 = None
        view_default_9: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_25, _shape_param_9);  sum_dim_int_list_25 = _shape_param_9 = None
        mul_tensor_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_26, arg108_1);  arg108_1 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_26, [0, 1]);  add_26 = None
        mul_tensor_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_27, arg107_1);  arg107_1 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_27, [0, 1]);  add_27 = None
        permute_default_9: "f32[768, 512]" = torch.ops.aten.permute.default(view_36, [1, 0])
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_36, [0], True);  view_36 = None
        view_default_10: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_30, _shape_param_10);  sum_dim_int_list_30 = _shape_param_10 = None
        permute_default_10: "f32[3072, 512]" = torch.ops.aten.permute.default(view_40, [1, 0])
        sum_dim_int_list_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_default_11: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_31, _shape_param_11);  sum_dim_int_list_31 = _shape_param_11 = None
        mul_tensor_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_32, arg102_1);  arg102_1 = None
        sum_dim_int_list_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1]);  add_32 = None
        mul_tensor_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_33, arg101_1);  arg101_1 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_33, [0, 1]);  add_33 = None
        permute_default_11: "f32[768, 512]" = torch.ops.aten.permute.default(view_43, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
        view_default_12: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_36, _shape_param_12);  sum_dim_int_list_36 = _shape_param_12 = None
        permute_default_12: "f32[3072, 512]" = torch.ops.aten.permute.default(view_47, [1, 0])
        sum_dim_int_list_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
        view_default_13: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_37, _shape_param_13);  sum_dim_int_list_37 = _shape_param_13 = None
        mul_tensor_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_38, arg96_1);  arg96_1 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_38, [0, 1]);  add_38 = None
        mul_tensor_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_39, arg95_1);  arg95_1 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_39, [0, 1]);  add_39 = None
        permute_default_13: "f32[768, 512]" = torch.ops.aten.permute.default(view_50, [1, 0])
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_50, [0], True);  view_50 = None
        view_default_14: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_42, _shape_param_14);  sum_dim_int_list_42 = _shape_param_14 = None
        permute_default_14: "f32[3072, 512]" = torch.ops.aten.permute.default(view_54, [1, 0])
        sum_dim_int_list_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_54, [0], True);  view_54 = None
        view_default_15: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_43, _shape_param_15);  sum_dim_int_list_43 = _shape_param_15 = None
        mul_tensor_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_44, arg90_1);  arg90_1 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_44, [0, 1]);  add_44 = None
        mul_tensor_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_45, arg89_1);  arg89_1 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_45, [0, 1]);  add_45 = None
        permute_default_15: "f32[768, 512]" = torch.ops.aten.permute.default(view_57, [1, 0])
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True);  view_57 = None
        view_default_16: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_48, _shape_param_16);  sum_dim_int_list_48 = _shape_param_16 = None
        permute_default_16: "f32[3072, 512]" = torch.ops.aten.permute.default(view_61, [1, 0])
        sum_dim_int_list_49: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True);  view_61 = None
        view_default_17: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_49, _shape_param_17);  sum_dim_int_list_49 = _shape_param_17 = None
        mul_tensor_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_50, arg84_1);  arg84_1 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_50, [0, 1]);  add_50 = None
        mul_tensor_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_51, arg83_1);  arg83_1 = None
        sum_dim_int_list_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_51, [0, 1]);  add_51 = None
        permute_default_17: "f32[768, 512]" = torch.ops.aten.permute.default(view_64, [1, 0])
        sum_dim_int_list_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True);  view_64 = None
        view_default_18: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_54, _shape_param_18);  sum_dim_int_list_54 = _shape_param_18 = None
        permute_default_18: "f32[3072, 512]" = torch.ops.aten.permute.default(view_68, [1, 0])
        sum_dim_int_list_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True);  view_68 = None
        view_default_19: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_55, _shape_param_19);  sum_dim_int_list_55 = _shape_param_19 = None
        mul_tensor_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_56, arg78_1);  arg78_1 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
        mul_tensor_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_57, arg77_1);  arg77_1 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_57, [0, 1]);  add_57 = None
        permute_default_19: "f32[768, 512]" = torch.ops.aten.permute.default(view_71, [1, 0])
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True);  view_71 = None
        view_default_20: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_60, _shape_param_20);  sum_dim_int_list_60 = _shape_param_20 = None
        permute_default_20: "f32[3072, 512]" = torch.ops.aten.permute.default(view_75, [1, 0])
        sum_dim_int_list_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        view_default_21: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_61, _shape_param_21);  sum_dim_int_list_61 = _shape_param_21 = None
        mul_tensor_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_62, arg72_1);  arg72_1 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_62, [0, 1]);  add_62 = None
        mul_tensor_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_63, arg71_1);  arg71_1 = None
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_63, [0, 1]);  add_63 = None
        permute_default_21: "f32[768, 512]" = torch.ops.aten.permute.default(view_78, [1, 0])
        sum_dim_int_list_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True);  view_78 = None
        view_default_22: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_66, _shape_param_22);  sum_dim_int_list_66 = _shape_param_22 = None
        permute_default_22: "f32[3072, 512]" = torch.ops.aten.permute.default(view_82, [1, 0])
        sum_dim_int_list_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_82, [0], True);  view_82 = None
        view_default_23: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_67, _shape_param_23);  sum_dim_int_list_67 = _shape_param_23 = None
        mul_tensor_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_68, arg66_1);  arg66_1 = None
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1]);  add_68 = None
        mul_tensor_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_69, arg65_1);  arg65_1 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_69, [0, 1]);  add_69 = None
        permute_default_23: "f32[768, 512]" = torch.ops.aten.permute.default(view_85, [1, 0])
        sum_dim_int_list_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_85, [0], True);  view_85 = None
        view_default_24: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_72, _shape_param_24);  sum_dim_int_list_72 = _shape_param_24 = None
        permute_default_24: "f32[3072, 512]" = torch.ops.aten.permute.default(view_89, [1, 0])
        sum_dim_int_list_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_89, [0], True);  view_89 = None
        view_default_25: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list_73, _shape_param_25);  sum_dim_int_list_73 = _shape_param_25 = None
        mul_tensor_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_74, arg60_1);  arg60_1 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_74, [0, 1]);  add_74 = None
        permute_default_25: "f32[768, 512]" = torch.ops.aten.permute.default(view_92, [1, 0])
        sum_dim_int_list_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_92, [0], True);  view_92 = None
        view_default_26: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_76, _shape_param_26);  sum_dim_int_list_76 = _shape_param_26 = None
        view_default_27: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, _shape_param_27);  mm_52 = _shape_param_27 = None
        mul_tensor_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_default_27, arg4_1);  arg4_1 = None
        mul_tensor_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_25, 768)
        sum_dim_int_list_77: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [2], True)
        mul_tensor_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_25, arg57_1);  mul_tensor_25 = None
        sum_dim_int_list_78: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True);  mul_tensor_27 = None
        mul_tensor_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg57_1, sum_dim_int_list_78);  sum_dim_int_list_78 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_26, sum_dim_int_list_77);  mul_tensor_26 = sum_dim_int_list_77 = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_28);  sub_tensor = mul_tensor_28 = None
        mul_tensor_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg165_1, sub_tensor_1);  arg165_1 = sub_tensor_1 = None
        mul_tensor_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_default_27, arg57_1);  arg57_1 = None
        sum_dim_int_list_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_default_27, [0, 1]);  view_default_27 = None
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg2_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_1, mul_tensor_29);  unsqueeze_default = None
        full_default: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default, [arg2_1], where_self, True);  full_default = arg2_1 = where_self = None
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(arg1_1, _shape_param_28);  arg1_1 = _shape_param_28 = None
        eq_scalar_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_29);  unsqueeze_default_1 = None
        full_default_1: "f32[4, 768]" = torch.ops.aten.full.default([4, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[4, 768]" = torch.ops.aten.index_put.default(full_default_1, [expand_default], where_self_1, True);  full_default_1 = expand_default = where_self_1 = None
        eq_scalar_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg0_1, 3)
        unsqueeze_default_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_1, mul_tensor_29);  unsqueeze_default_2 = full_1 = mul_tensor_29 = None
        full_default_2: "f32[32000, 768]" = torch.ops.aten.full.default([32000, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[32000, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg0_1], where_self_2, True);  full_default_2 = arg0_1 = where_self_2 = None
        add_tensor: "f32[32000, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (view_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, view_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, view_default_2, permute_default_2, view_default_3, sum_dim_int_list_8, sum_dim_int_list_9, sum_dim_int_list_10, sum_dim_int_list_11, permute_default_3, view_default_4, permute_default_4, view_default_5, sum_dim_int_list_14, sum_dim_int_list_15, sum_dim_int_list_16, sum_dim_int_list_17, permute_default_5, view_default_6, permute_default_6, view_default_7, sum_dim_int_list_20, sum_dim_int_list_21, sum_dim_int_list_22, sum_dim_int_list_23, permute_default_7, view_default_8, permute_default_8, view_default_9, sum_dim_int_list_26, sum_dim_int_list_27, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_9, view_default_10, permute_default_10, view_default_11, sum_dim_int_list_32, sum_dim_int_list_33, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_11, view_default_12, permute_default_12, view_default_13, sum_dim_int_list_38, sum_dim_int_list_39, sum_dim_int_list_40, sum_dim_int_list_41, permute_default_13, view_default_14, permute_default_14, view_default_15, sum_dim_int_list_44, sum_dim_int_list_45, sum_dim_int_list_46, sum_dim_int_list_47, permute_default_15, view_default_16, permute_default_16, view_default_17, sum_dim_int_list_50, sum_dim_int_list_51, sum_dim_int_list_52, sum_dim_int_list_53, permute_default_17, view_default_18, permute_default_18, view_default_19, sum_dim_int_list_56, sum_dim_int_list_57, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_19, view_default_20, permute_default_20, view_default_21, sum_dim_int_list_62, sum_dim_int_list_63, sum_dim_int_list_64, sum_dim_int_list_65, permute_default_21, view_default_22, permute_default_22, view_default_23, sum_dim_int_list_68, sum_dim_int_list_69, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_23, view_default_24, permute_default_24, view_default_25, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_25, view_default_26, sum_dim_int_list_79, sum_dim_int_list_80, index_put_default, index_put_default_1, add_tensor)


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
