"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g53
Pattern hash: 9292de0b9be9
Shape hash: ab7baff1
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
    def forward(self, arg297_1: "f16[128, 1000]", mm_1: "f16[1000, 1408]", sum_3: "f32[1408]", squeeze_1: "f32[1408]", getitem_1: "f16[1408, 384, 3, 3]", sum_5: "f32[1408]", squeeze_3: "f32[1408]", getitem_4: "f16[1408, 384, 1, 1]", sum_7: "f32[384]", arg226_1: "f32[384]", getitem_7: "f16[384, 384, 3, 3]", sum_9: "f32[384]", arg223_1: "f32[384]", getitem_10: "f16[384, 384, 1, 1]", sum_11: "f32[384]", arg220_1: "f32[384]", sum_13: "f32[384]", arg218_1: "f32[384]", getitem_13: "f16[384, 384, 3, 3]", sum_15: "f32[384]", arg215_1: "f32[384]", getitem_16: "f16[384, 384, 1, 1]", sum_17: "f32[384]", arg212_1: "f32[384]", sum_19: "f32[384]", arg210_1: "f32[384]", getitem_19: "f16[384, 384, 3, 3]", sum_21: "f32[384]", arg207_1: "f32[384]", getitem_22: "f16[384, 384, 1, 1]", sum_23: "f32[384]", arg204_1: "f32[384]", sum_25: "f32[384]", arg202_1: "f32[384]", getitem_25: "f16[384, 384, 3, 3]", sum_27: "f32[384]", arg199_1: "f32[384]", getitem_28: "f16[384, 384, 1, 1]", sum_29: "f32[384]", arg196_1: "f32[384]", sum_31: "f32[384]", arg194_1: "f32[384]", getitem_31: "f16[384, 384, 3, 3]", sum_33: "f32[384]", arg191_1: "f32[384]", getitem_34: "f16[384, 384, 1, 1]", sum_35: "f32[384]", arg188_1: "f32[384]", sum_37: "f32[384]", arg186_1: "f32[384]", getitem_37: "f16[384, 384, 3, 3]", sum_39: "f32[384]", arg183_1: "f32[384]", getitem_40: "f16[384, 384, 1, 1]", sum_41: "f32[384]", arg180_1: "f32[384]", sum_43: "f32[384]", arg178_1: "f32[384]", getitem_43: "f16[384, 384, 3, 3]", sum_45: "f32[384]", arg175_1: "f32[384]", getitem_46: "f16[384, 384, 1, 1]", sum_47: "f32[384]", arg172_1: "f32[384]", sum_49: "f32[384]", arg170_1: "f32[384]", getitem_49: "f16[384, 384, 3, 3]", sum_51: "f32[384]", arg167_1: "f32[384]", getitem_52: "f16[384, 384, 1, 1]", sum_53: "f32[384]", arg164_1: "f32[384]", sum_55: "f32[384]", arg162_1: "f32[384]", getitem_55: "f16[384, 384, 3, 3]", sum_57: "f32[384]", arg159_1: "f32[384]", getitem_58: "f16[384, 384, 1, 1]", sum_59: "f32[384]", arg156_1: "f32[384]", sum_61: "f32[384]", arg154_1: "f32[384]", getitem_61: "f16[384, 384, 3, 3]", sum_63: "f32[384]", arg151_1: "f32[384]", getitem_64: "f16[384, 384, 1, 1]", sum_65: "f32[384]", arg148_1: "f32[384]", sum_67: "f32[384]", arg146_1: "f32[384]", getitem_67: "f16[384, 384, 3, 3]", sum_69: "f32[384]", arg143_1: "f32[384]", getitem_70: "f16[384, 384, 1, 1]", sum_71: "f32[384]", arg140_1: "f32[384]", sum_73: "f32[384]", arg138_1: "f32[384]", getitem_73: "f16[384, 384, 3, 3]", sum_75: "f32[384]", arg135_1: "f32[384]", getitem_76: "f16[384, 384, 1, 1]", sum_77: "f32[384]", arg132_1: "f32[384]", sum_79: "f32[384]", arg130_1: "f32[384]", getitem_79: "f16[384, 384, 3, 3]", sum_81: "f32[384]", arg127_1: "f32[384]", getitem_82: "f16[384, 384, 1, 1]", sum_83: "f32[384]", arg124_1: "f32[384]", sum_85: "f32[384]", arg122_1: "f32[384]", getitem_85: "f16[384, 192, 3, 3]", sum_87: "f32[384]", arg119_1: "f32[384]", getitem_88: "f16[384, 192, 1, 1]", sum_89: "f32[192]", arg115_1: "f32[192]", getitem_91: "f16[192, 192, 3, 3]", sum_91: "f32[192]", arg112_1: "f32[192]", getitem_94: "f16[192, 192, 1, 1]", sum_93: "f32[192]", arg109_1: "f32[192]", sum_95: "f32[192]", arg107_1: "f32[192]", getitem_97: "f16[192, 192, 3, 3]", sum_97: "f32[192]", arg104_1: "f32[192]", getitem_100: "f16[192, 192, 1, 1]", sum_99: "f32[192]", arg101_1: "f32[192]", sum_101: "f32[192]", arg99_1: "f32[192]", getitem_103: "f16[192, 192, 3, 3]", sum_103: "f32[192]", arg96_1: "f32[192]", getitem_106: "f16[192, 192, 1, 1]", sum_105: "f32[192]", arg93_1: "f32[192]", sum_107: "f32[192]", arg91_1: "f32[192]", getitem_109: "f16[192, 96, 3, 3]", sum_109: "f32[192]", arg88_1: "f32[192]", getitem_112: "f16[192, 96, 1, 1]", sum_111: "f32[96]", arg84_1: "f32[96]", getitem_115: "f16[96, 96, 3, 3]", sum_113: "f32[96]", arg81_1: "f32[96]", getitem_118: "f16[96, 96, 1, 1]", sum_115: "f32[96]", arg78_1: "f32[96]", sum_117: "f32[96]", arg76_1: "f32[96]", getitem_121: "f16[96, 64, 3, 3]", sum_119: "f32[96]", arg73_1: "f32[96]", getitem_124: "f16[96, 64, 1, 1]", sum_121: "f32[64]", arg69_1: "f32[64]", getitem_127: "f16[64, 3, 3, 3]", sum_123: "f32[64]", arg66_1: "f32[64]", getitem_130: "f16[64, 3, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f16[1, 1000]" = torch.ops.aten.sum.dim_IntList(arg297_1, [0], True);  arg297_1 = None
        reshape_default: "f16[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        convert_element_type_default: "f32[1000, 1408]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_1: "f32[1000]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_1);  sum_3 = squeeze_1 = None
        convert_element_type_default_2: "f32[1408, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        mul_tensor_1: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_3);  sum_5 = squeeze_3 = None
        convert_element_type_default_3: "f32[1408, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_4, torch.float32);  getitem_4 = None
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, arg226_1);  sum_7 = arg226_1 = None
        convert_element_type_default_4: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, arg223_1);  sum_9 = arg223_1 = None
        convert_element_type_default_5: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(sum_11, arg220_1);  sum_11 = arg220_1 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, arg218_1);  sum_13 = arg218_1 = None
        convert_element_type_default_6: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, arg215_1);  sum_15 = arg215_1 = None
        convert_element_type_default_7: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        mul_tensor_7: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, arg212_1);  sum_17 = arg212_1 = None
        mul_tensor_8: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, arg210_1);  sum_19 = arg210_1 = None
        convert_element_type_default_8: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(sum_21, arg207_1);  sum_21 = arg207_1 = None
        convert_element_type_default_9: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, arg204_1);  sum_23 = arg204_1 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, arg202_1);  sum_25 = arg202_1 = None
        convert_element_type_default_10: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, arg199_1);  sum_27 = arg199_1 = None
        convert_element_type_default_11: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(sum_29, arg196_1);  sum_29 = arg196_1 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, arg194_1);  sum_31 = arg194_1 = None
        convert_element_type_default_12: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, arg191_1);  sum_33 = arg191_1 = None
        convert_element_type_default_13: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, arg188_1);  sum_35 = arg188_1 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(sum_37, arg186_1);  sum_37 = arg186_1 = None
        convert_element_type_default_14: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, arg183_1);  sum_39 = arg183_1 = None
        convert_element_type_default_15: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
        mul_tensor_19: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, arg180_1);  sum_41 = arg180_1 = None
        mul_tensor_20: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, arg178_1);  sum_43 = arg178_1 = None
        convert_element_type_default_16: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
        mul_tensor_21: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, arg175_1);  sum_45 = arg175_1 = None
        convert_element_type_default_17: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_46, torch.float32);  getitem_46 = None
        mul_tensor_22: "f32[384]" = torch.ops.aten.mul.Tensor(sum_47, arg172_1);  sum_47 = arg172_1 = None
        mul_tensor_23: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, arg170_1);  sum_49 = arg170_1 = None
        convert_element_type_default_18: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
        mul_tensor_24: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, arg167_1);  sum_51 = arg167_1 = None
        convert_element_type_default_19: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None
        mul_tensor_25: "f32[384]" = torch.ops.aten.mul.Tensor(sum_53, arg164_1);  sum_53 = arg164_1 = None
        mul_tensor_26: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, arg162_1);  sum_55 = arg162_1 = None
        convert_element_type_default_20: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
        mul_tensor_27: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, arg159_1);  sum_57 = arg159_1 = None
        convert_element_type_default_21: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        mul_tensor_28: "f32[384]" = torch.ops.aten.mul.Tensor(sum_59, arg156_1);  sum_59 = arg156_1 = None
        mul_tensor_29: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, arg154_1);  sum_61 = arg154_1 = None
        convert_element_type_default_22: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
        mul_tensor_30: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, arg151_1);  sum_63 = arg151_1 = None
        convert_element_type_default_23: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_64, torch.float32);  getitem_64 = None
        mul_tensor_31: "f32[384]" = torch.ops.aten.mul.Tensor(sum_65, arg148_1);  sum_65 = arg148_1 = None
        mul_tensor_32: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, arg146_1);  sum_67 = arg146_1 = None
        convert_element_type_default_24: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
        mul_tensor_33: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, arg143_1);  sum_69 = arg143_1 = None
        convert_element_type_default_25: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_70, torch.float32);  getitem_70 = None
        mul_tensor_34: "f32[384]" = torch.ops.aten.mul.Tensor(sum_71, arg140_1);  sum_71 = arg140_1 = None
        mul_tensor_35: "f32[384]" = torch.ops.aten.mul.Tensor(sum_73, arg138_1);  sum_73 = arg138_1 = None
        convert_element_type_default_26: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
        mul_tensor_36: "f32[384]" = torch.ops.aten.mul.Tensor(sum_75, arg135_1);  sum_75 = arg135_1 = None
        convert_element_type_default_27: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_76, torch.float32);  getitem_76 = None
        mul_tensor_37: "f32[384]" = torch.ops.aten.mul.Tensor(sum_77, arg132_1);  sum_77 = arg132_1 = None
        mul_tensor_38: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, arg130_1);  sum_79 = arg130_1 = None
        convert_element_type_default_28: "f32[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
        mul_tensor_39: "f32[384]" = torch.ops.aten.mul.Tensor(sum_81, arg127_1);  sum_81 = arg127_1 = None
        convert_element_type_default_29: "f32[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None
        mul_tensor_40: "f32[384]" = torch.ops.aten.mul.Tensor(sum_83, arg124_1);  sum_83 = arg124_1 = None
        mul_tensor_41: "f32[384]" = torch.ops.aten.mul.Tensor(sum_85, arg122_1);  sum_85 = arg122_1 = None
        convert_element_type_default_30: "f32[384, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
        mul_tensor_42: "f32[384]" = torch.ops.aten.mul.Tensor(sum_87, arg119_1);  sum_87 = arg119_1 = None
        convert_element_type_default_31: "f32[384, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_88, torch.float32);  getitem_88 = None
        mul_tensor_43: "f32[192]" = torch.ops.aten.mul.Tensor(sum_89, arg115_1);  sum_89 = arg115_1 = None
        convert_element_type_default_32: "f32[192, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
        mul_tensor_44: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, arg112_1);  sum_91 = arg112_1 = None
        convert_element_type_default_33: "f32[192, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_94, torch.float32);  getitem_94 = None
        mul_tensor_45: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, arg109_1);  sum_93 = arg109_1 = None
        mul_tensor_46: "f32[192]" = torch.ops.aten.mul.Tensor(sum_95, arg107_1);  sum_95 = arg107_1 = None
        convert_element_type_default_34: "f32[192, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
        mul_tensor_47: "f32[192]" = torch.ops.aten.mul.Tensor(sum_97, arg104_1);  sum_97 = arg104_1 = None
        convert_element_type_default_35: "f32[192, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_100, torch.float32);  getitem_100 = None
        mul_tensor_48: "f32[192]" = torch.ops.aten.mul.Tensor(sum_99, arg101_1);  sum_99 = arg101_1 = None
        mul_tensor_49: "f32[192]" = torch.ops.aten.mul.Tensor(sum_101, arg99_1);  sum_101 = arg99_1 = None
        convert_element_type_default_36: "f32[192, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
        mul_tensor_50: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, arg96_1);  sum_103 = arg96_1 = None
        convert_element_type_default_37: "f32[192, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_106, torch.float32);  getitem_106 = None
        mul_tensor_51: "f32[192]" = torch.ops.aten.mul.Tensor(sum_105, arg93_1);  sum_105 = arg93_1 = None
        mul_tensor_52: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, arg91_1);  sum_107 = arg91_1 = None
        convert_element_type_default_38: "f32[192, 96, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None
        mul_tensor_53: "f32[192]" = torch.ops.aten.mul.Tensor(sum_109, arg88_1);  sum_109 = arg88_1 = None
        convert_element_type_default_39: "f32[192, 96, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32);  getitem_112 = None
        mul_tensor_54: "f32[96]" = torch.ops.aten.mul.Tensor(sum_111, arg84_1);  sum_111 = arg84_1 = None
        convert_element_type_default_40: "f32[96, 96, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
        mul_tensor_55: "f32[96]" = torch.ops.aten.mul.Tensor(sum_113, arg81_1);  sum_113 = arg81_1 = None
        convert_element_type_default_41: "f32[96, 96, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32);  getitem_118 = None
        mul_tensor_56: "f32[96]" = torch.ops.aten.mul.Tensor(sum_115, arg78_1);  sum_115 = arg78_1 = None
        mul_tensor_57: "f32[96]" = torch.ops.aten.mul.Tensor(sum_117, arg76_1);  sum_117 = arg76_1 = None
        convert_element_type_default_42: "f32[96, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
        mul_tensor_58: "f32[96]" = torch.ops.aten.mul.Tensor(sum_119, arg73_1);  sum_119 = arg73_1 = None
        convert_element_type_default_43: "f32[96, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_124, torch.float32);  getitem_124 = None
        mul_tensor_59: "f32[64]" = torch.ops.aten.mul.Tensor(sum_121, arg69_1);  sum_121 = arg69_1 = None
        convert_element_type_default_44: "f32[64, 3, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
        mul_tensor_60: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, arg66_1);  sum_123 = arg66_1 = None
        convert_element_type_default_45: "f32[64, 3, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_130, torch.float32);  getitem_130 = None
        _output_to_half_0: "f16[1000, 1408]" = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float16);  convert_element_type_default = None
        _output_to_half_1: "f16[1000]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float16);  convert_element_type_default_1 = None
        _output_to_half_2: "f16[1408]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        _output_to_half_3: "f16[1408, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float16);  convert_element_type_default_2 = None
        _output_to_half_4: "f16[1408]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        _output_to_half_5: "f16[1408, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float16);  convert_element_type_default_3 = None
        _output_to_half_6: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None
        _output_to_half_7: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float16);  convert_element_type_default_4 = None
        _output_to_half_8: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        _output_to_half_9: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float16);  convert_element_type_default_5 = None
        _output_to_half_10: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float16);  mul_tensor_4 = None
        _output_to_half_11: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.float16);  mul_tensor_5 = None
        _output_to_half_12: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float16);  convert_element_type_default_6 = None
        _output_to_half_13: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.float16);  mul_tensor_6 = None
        _output_to_half_14: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_7, torch.float16);  convert_element_type_default_7 = None
        _output_to_half_15: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float16);  mul_tensor_7 = None
        _output_to_half_16: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float16);  mul_tensor_8 = None
        _output_to_half_17: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_8, torch.float16);  convert_element_type_default_8 = None
        _output_to_half_18: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.float16);  mul_tensor_9 = None
        _output_to_half_19: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_9, torch.float16);  convert_element_type_default_9 = None
        _output_to_half_20: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.float16);  mul_tensor_10 = None
        _output_to_half_21: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.float16);  mul_tensor_11 = None
        _output_to_half_22: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_10, torch.float16);  convert_element_type_default_10 = None
        _output_to_half_23: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float16);  mul_tensor_12 = None
        _output_to_half_24: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_11, torch.float16);  convert_element_type_default_11 = None
        _output_to_half_25: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.float16);  mul_tensor_13 = None
        _output_to_half_26: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.float16);  mul_tensor_14 = None
        _output_to_half_27: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_12, torch.float16);  convert_element_type_default_12 = None
        _output_to_half_28: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_15, torch.float16);  mul_tensor_15 = None
        _output_to_half_29: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_13, torch.float16);  convert_element_type_default_13 = None
        _output_to_half_30: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.float16);  mul_tensor_16 = None
        _output_to_half_31: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_17, torch.float16);  mul_tensor_17 = None
        _output_to_half_32: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_14, torch.float16);  convert_element_type_default_14 = None
        _output_to_half_33: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.float16);  mul_tensor_18 = None
        _output_to_half_34: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_15, torch.float16);  convert_element_type_default_15 = None
        _output_to_half_35: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_19, torch.float16);  mul_tensor_19 = None
        _output_to_half_36: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.float16);  mul_tensor_20 = None
        _output_to_half_37: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_16, torch.float16);  convert_element_type_default_16 = None
        _output_to_half_38: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_21, torch.float16);  mul_tensor_21 = None
        _output_to_half_39: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_17, torch.float16);  convert_element_type_default_17 = None
        _output_to_half_40: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_22, torch.float16);  mul_tensor_22 = None
        _output_to_half_41: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_23, torch.float16);  mul_tensor_23 = None
        _output_to_half_42: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_18, torch.float16);  convert_element_type_default_18 = None
        _output_to_half_43: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.float16);  mul_tensor_24 = None
        _output_to_half_44: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_19, torch.float16);  convert_element_type_default_19 = None
        _output_to_half_45: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_25, torch.float16);  mul_tensor_25 = None
        _output_to_half_46: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_26, torch.float16);  mul_tensor_26 = None
        _output_to_half_47: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_20, torch.float16);  convert_element_type_default_20 = None
        _output_to_half_48: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_27, torch.float16);  mul_tensor_27 = None
        _output_to_half_49: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_21, torch.float16);  convert_element_type_default_21 = None
        _output_to_half_50: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.float16);  mul_tensor_28 = None
        _output_to_half_51: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_29, torch.float16);  mul_tensor_29 = None
        _output_to_half_52: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_22, torch.float16);  convert_element_type_default_22 = None
        _output_to_half_53: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_30, torch.float16);  mul_tensor_30 = None
        _output_to_half_54: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_23, torch.float16);  convert_element_type_default_23 = None
        _output_to_half_55: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_31, torch.float16);  mul_tensor_31 = None
        _output_to_half_56: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.float16);  mul_tensor_32 = None
        _output_to_half_57: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_24, torch.float16);  convert_element_type_default_24 = None
        _output_to_half_58: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_33, torch.float16);  mul_tensor_33 = None
        _output_to_half_59: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_25, torch.float16);  convert_element_type_default_25 = None
        _output_to_half_60: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_34, torch.float16);  mul_tensor_34 = None
        _output_to_half_61: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_35, torch.float16);  mul_tensor_35 = None
        _output_to_half_62: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_26, torch.float16);  convert_element_type_default_26 = None
        _output_to_half_63: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.float16);  mul_tensor_36 = None
        _output_to_half_64: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_27, torch.float16);  convert_element_type_default_27 = None
        _output_to_half_65: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_37, torch.float16);  mul_tensor_37 = None
        _output_to_half_66: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_38, torch.float16);  mul_tensor_38 = None
        _output_to_half_67: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_28, torch.float16);  convert_element_type_default_28 = None
        _output_to_half_68: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_39, torch.float16);  mul_tensor_39 = None
        _output_to_half_69: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_29, torch.float16);  convert_element_type_default_29 = None
        _output_to_half_70: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.float16);  mul_tensor_40 = None
        _output_to_half_71: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_41, torch.float16);  mul_tensor_41 = None
        _output_to_half_72: "f16[384, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_30, torch.float16);  convert_element_type_default_30 = None
        _output_to_half_73: "f16[384]" = torch.ops.prims.convert_element_type.default(mul_tensor_42, torch.float16);  mul_tensor_42 = None
        _output_to_half_74: "f16[384, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_31, torch.float16);  convert_element_type_default_31 = None
        _output_to_half_75: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_43, torch.float16);  mul_tensor_43 = None
        _output_to_half_76: "f16[192, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_32, torch.float16);  convert_element_type_default_32 = None
        _output_to_half_77: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.float16);  mul_tensor_44 = None
        _output_to_half_78: "f16[192, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_33, torch.float16);  convert_element_type_default_33 = None
        _output_to_half_79: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_45, torch.float16);  mul_tensor_45 = None
        _output_to_half_80: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_46, torch.float16);  mul_tensor_46 = None
        _output_to_half_81: "f16[192, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_34, torch.float16);  convert_element_type_default_34 = None
        _output_to_half_82: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_47, torch.float16);  mul_tensor_47 = None
        _output_to_half_83: "f16[192, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_35, torch.float16);  convert_element_type_default_35 = None
        _output_to_half_84: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_48, torch.float16);  mul_tensor_48 = None
        _output_to_half_85: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_49, torch.float16);  mul_tensor_49 = None
        _output_to_half_86: "f16[192, 192, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_36, torch.float16);  convert_element_type_default_36 = None
        _output_to_half_87: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_50, torch.float16);  mul_tensor_50 = None
        _output_to_half_88: "f16[192, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_37, torch.float16);  convert_element_type_default_37 = None
        _output_to_half_89: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_51, torch.float16);  mul_tensor_51 = None
        _output_to_half_90: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.float16);  mul_tensor_52 = None
        _output_to_half_91: "f16[192, 96, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_38, torch.float16);  convert_element_type_default_38 = None
        _output_to_half_92: "f16[192]" = torch.ops.prims.convert_element_type.default(mul_tensor_53, torch.float16);  mul_tensor_53 = None
        _output_to_half_93: "f16[192, 96, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_39, torch.float16);  convert_element_type_default_39 = None
        _output_to_half_94: "f16[96]" = torch.ops.prims.convert_element_type.default(mul_tensor_54, torch.float16);  mul_tensor_54 = None
        _output_to_half_95: "f16[96, 96, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_40, torch.float16);  convert_element_type_default_40 = None
        _output_to_half_96: "f16[96]" = torch.ops.prims.convert_element_type.default(mul_tensor_55, torch.float16);  mul_tensor_55 = None
        _output_to_half_97: "f16[96, 96, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_41, torch.float16);  convert_element_type_default_41 = None
        _output_to_half_98: "f16[96]" = torch.ops.prims.convert_element_type.default(mul_tensor_56, torch.float16);  mul_tensor_56 = None
        _output_to_half_99: "f16[96]" = torch.ops.prims.convert_element_type.default(mul_tensor_57, torch.float16);  mul_tensor_57 = None
        _output_to_half_100: "f16[96, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_42, torch.float16);  convert_element_type_default_42 = None
        _output_to_half_101: "f16[96]" = torch.ops.prims.convert_element_type.default(mul_tensor_58, torch.float16);  mul_tensor_58 = None
        _output_to_half_102: "f16[96, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_43, torch.float16);  convert_element_type_default_43 = None
        _output_to_half_103: "f16[64]" = torch.ops.prims.convert_element_type.default(mul_tensor_59, torch.float16);  mul_tensor_59 = None
        _output_to_half_104: "f16[64, 3, 3, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_44, torch.float16);  convert_element_type_default_44 = None
        _output_to_half_105: "f16[64]" = torch.ops.prims.convert_element_type.default(mul_tensor_60, torch.float16);  mul_tensor_60 = None
        _output_to_half_106: "f16[64, 3, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_45, torch.float16);  convert_element_type_default_45 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3, _output_to_half_4, _output_to_half_5, _output_to_half_6, _output_to_half_7, _output_to_half_8, _output_to_half_9, _output_to_half_10, _output_to_half_11, _output_to_half_12, _output_to_half_13, _output_to_half_14, _output_to_half_15, _output_to_half_16, _output_to_half_17, _output_to_half_18, _output_to_half_19, _output_to_half_20, _output_to_half_21, _output_to_half_22, _output_to_half_23, _output_to_half_24, _output_to_half_25, _output_to_half_26, _output_to_half_27, _output_to_half_28, _output_to_half_29, _output_to_half_30, _output_to_half_31, _output_to_half_32, _output_to_half_33, _output_to_half_34, _output_to_half_35, _output_to_half_36, _output_to_half_37, _output_to_half_38, _output_to_half_39, _output_to_half_40, _output_to_half_41, _output_to_half_42, _output_to_half_43, _output_to_half_44, _output_to_half_45, _output_to_half_46, _output_to_half_47, _output_to_half_48, _output_to_half_49, _output_to_half_50, _output_to_half_51, _output_to_half_52, _output_to_half_53, _output_to_half_54, _output_to_half_55, _output_to_half_56, _output_to_half_57, _output_to_half_58, _output_to_half_59, _output_to_half_60, _output_to_half_61, _output_to_half_62, _output_to_half_63, _output_to_half_64, _output_to_half_65, _output_to_half_66, _output_to_half_67, _output_to_half_68, _output_to_half_69, _output_to_half_70, _output_to_half_71, _output_to_half_72, _output_to_half_73, _output_to_half_74, _output_to_half_75, _output_to_half_76, _output_to_half_77, _output_to_half_78, _output_to_half_79, _output_to_half_80, _output_to_half_81, _output_to_half_82, _output_to_half_83, _output_to_half_84, _output_to_half_85, _output_to_half_86, _output_to_half_87, _output_to_half_88, _output_to_half_89, _output_to_half_90, _output_to_half_91, _output_to_half_92, _output_to_half_93, _output_to_half_94, _output_to_half_95, _output_to_half_96, _output_to_half_97, _output_to_half_98, _output_to_half_99, _output_to_half_100, _output_to_half_101, _output_to_half_102, _output_to_half_103, _output_to_half_104, _output_to_half_105, _output_to_half_106)


def _default_make_inputs():
    return [
    torch.randn([128, 1000], dtype=torch.float16, device='cuda'),
    torch.randn([1000, 1408], dtype=torch.float16, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn(4866048, dtype=torch.float16, device='cuda').as_strided([1408, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_1
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_7
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_13
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_19
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_25
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_31
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_37
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_43
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_49
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_55
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_61
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_67
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_73
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1327104, dtype=torch.float16, device='cuda').as_strided([384, 384, 3, 3], [3456, 1, 1152, 384]),  # getitem_79
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(663552, dtype=torch.float16, device='cuda').as_strided([384, 192, 3, 3], [1728, 1, 576, 192]),  # getitem_85
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(331776, dtype=torch.float16, device='cuda').as_strided([192, 192, 3, 3], [1728, 1, 576, 192]),  # getitem_91
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(331776, dtype=torch.float16, device='cuda').as_strided([192, 192, 3, 3], [1728, 1, 576, 192]),  # getitem_97
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(331776, dtype=torch.float16, device='cuda').as_strided([192, 192, 3, 3], [1728, 1, 576, 192]),  # getitem_103
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(165888, dtype=torch.float16, device='cuda').as_strided([192, 96, 3, 3], [864, 1, 288, 96]),  # getitem_109
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(82944, dtype=torch.float16, device='cuda').as_strided([96, 96, 3, 3], [864, 1, 288, 96]),  # getitem_115
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(55296, dtype=torch.float16, device='cuda').as_strided([96, 64, 3, 3], [576, 1, 192, 64]),  # getitem_121
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(1728, dtype=torch.float16, device='cuda').as_strided([64, 3, 3, 3], [27, 1, 9, 3]),  # getitem_127
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 1, 1], dtype=torch.float16, device='cuda'),
    [1000],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
